# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import torchaudio
import torch
from loguru import logger

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加子模块路径
submodule_path = os.path.join(current_dir, '../', 'third_party/CosyVoice')
sys.path.insert(0, submodule_path)
matcha_tts_path = os.path.join(current_dir, '../', 'third_party/CosyVoice/third_party/Matcha-TTS')
sys.path.insert(0, matcha_tts_path)
import uuid
# 现在可以导入子模块中的库
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice3
from cosyvoice.utils.file_utils import load_wav


def get_prompt_from_wav(cosyvoice3_model, prompt_wav, max_prompt_sec=28):
    """Extract zero-shot voice-cloning conditioning from a reference wav so that
    token2wav renders the generated speech in that speaker's voice.

    Returns (embedding, prompt_token, prompt_feat) mirroring
    CosyVoiceFrontEnd.frontend_zero_shot: spk embedding (campplus), prompt speech
    tokens (speech_tokenizer_v3) and prompt mel feat, with the token_mel_ratio=2
    alignment (prompt_feat == 2 * prompt_token). The speech tokenizer hard-asserts
    <=30s, so a long reference is trimmed to the first max_prompt_sec seconds.
    """
    fe = cosyvoice3_model.frontend
    info = torchaudio.info(prompt_wav)
    src, tmp = prompt_wav, None
    if info.num_frames / info.sample_rate > max_prompt_sec:
        wav = load_wav(prompt_wav, 16000)[:, : int(max_prompt_sec * 16000)]
        tmp = f"{current_dir}/_prompt_{uuid.uuid4().hex}.wav"
        torchaudio.save(tmp, wav, 16000)
        src = tmp
    try:
        embedding = fe._extract_spk_embedding(src)
        speech_token, _ = fe._extract_speech_token(src)
        speech_feat, _ = fe._extract_speech_feat(src)
        token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
        speech_feat = speech_feat[:, : 2 * token_len]
        speech_token = speech_token[:, :token_len]
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    return embedding, speech_token.to(torch.int32), speech_feat


def token2wav(cosyvoice3_model, tokens, embedding=None, prompt_token=None, prompt_feat=None,
              token_hop_len=25 * 30, pre_lookahead_len=3):
    # logger.info(f"token2wav, num tokens: {len(tokens)}")
    if embedding is None:
        embedding = torch.load(f"{current_dir}/new_spk2info.pt")["中文女"]["embedding"]
    # No reference prompt -> empty (speaker comes from `embedding` alone). With a
    # prompt (from get_prompt_from_wav) the flow zero-shot-clones that voice.
    if prompt_token is None:
        prompt_token = torch.zeros(1, 0, dtype=torch.int32)
    if prompt_feat is None:
        prompt_feat = torch.zeros(1, 0, 80)

    speech_list = []

    time_step = 0
    tokens_list = []

    # Step 1: Split audio into 30-second segments
    while time_step * 25 < len(tokens):
        start = time_step * 25
        end = min((time_step + 30) * 25, len(tokens))
        token_segment = tokens[start:end]
        tokens_list.append(token_segment)
        time_step += 30

    # Step 2: Handle last segment if too short
    if len(tokens_list) > 1 and len(tokens_list[-1]) < 50:  # Less than 2 second
        # Remove last two segments
        last_segment = tokens_list.pop()
        second_last_segment = tokens_list.pop()

        # Merge last two segments
        merged_audio = second_last_segment + last_segment
        total_length = len(merged_audio)

        # Split merged audio into two equal parts
        split_point = total_length // 2
        first_half = merged_audio[:split_point]
        second_half = merged_audio[split_point:]

        # Add new segments back to list
        tokens_list.append(first_half)
        tokens_list.append(second_half)
    for token_segment in tokens_list:
        this_uuid = str(uuid.uuid4())
        cosyvoice3_model.model.hift_cache_dict[this_uuid] = None
        token_offset = 0
        for i in range(0, len(token_segment), token_hop_len):
            this_token = torch.tensor(token_segment[: token_offset + token_hop_len + pre_lookahead_len]).view(1, -1)
            finalize = True if this_token.shape[1] == len(token_segment) else False
            this_speech = cosyvoice3_model.model.token2wav(this_token, prompt_token,
                                                           prompt_feat, embedding, token_offset, this_uuid,
                                                           stream=False, finalize=finalize, speed=1.0)
            speech_list.append(this_speech)
            token_offset += token_hop_len
        del cosyvoice3_model.model.hift_cache_dict[this_uuid]
    speech = torch.concat(speech_list, dim=1)
    return  speech

def get_audio_detokenizer(token_hop_len=25 * 30):
    logger.info(f"cosyvoice cuda: {torch.cuda.is_available()}")
    cosyvoice3 = CosyVoice3('pretrained_models/Fun-CosyVoice3-0.5B-2512',
                            load_trt=False, load_vllm=False, fp16=False)
    cosyvoice3.model.flow.decoder.estimator.static_chunk_size = 2 * token_hop_len
    logger.info(f"cosyvoice loaded")
    return cosyvoice3

def tts_infer_streaming(tts_model, tts_spk_embedding, tokens, offset, uuid, finalize=False, token_hop_len=5, pre_lookahead_len=3, device="cuda:0"):
    # 确保所有 tensor 都在同一个 device 上
    speech = tts_model.model.token2wav(
        tokens, 
        torch.zeros(1, 0, dtype=torch.int32, device=device), 
        torch.zeros(1, 0, 80, device=device), 
        tts_spk_embedding, 
        offset, 
        uuid, 
        stream=False, 
        finalize=finalize, 
        speed=1.0
    )
    return speech