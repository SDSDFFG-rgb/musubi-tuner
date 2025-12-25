import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 最新版のLyCORISを使用
import lycoris.kohya as lycoris_network

# 最新版LyCORISでは config.py 内で自動認識されるため、
# 手動でのモジュール指定は不要になりました。
# ただし、将来的な互換性や明示的な指定のために変数定義だけ残しても害はありません。
# ZIMAGE_TARGET_REPLACE_MODULES = ["ZImageTransformerBlock"] 

def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    # 除外パターンの設定
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        if isinstance(exclude_patterns, str):
            exclude_patterns = ast.literal_eval(exclude_patterns)

    # modulationやrefinerの除外設定は維持
    exclude_patterns.append(r".*(_modulation|_refiner).*")
    kwargs["exclude_patterns"] = exclude_patterns

    # ---------------------------------------------------------
    # 【変更点】
    # LyCORIS v3.4.0+ で ZImageTransformerBlock がネイティブサポートされたため
    # 手動での target_module / target_name の注入コードは削除しました。
    # ---------------------------------------------------------
    
    # 念のため、過去のスクリプトで混入した不要な設定があればクリアすることも検討できますが、
    # 基本的にはそのまま渡してLyCORIS側のデフォルト動作に任せます。

    return lycoris_network.create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )

def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    return lycoris_network.create_network_from_weights(
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference=for_inference,
        **kwargs
    )
