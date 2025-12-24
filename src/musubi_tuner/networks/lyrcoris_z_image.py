import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 変更点1: 標準のloraではなくlycoris.kohyaをインポート
# 事前に `pip install lycoris-lora` が必要です
import lycoris.kohya as lycoris_network

# LyCORISは通常UNet全体を走査するため、明示的なターゲット指定は
# lycoris生成時には必須ではありませんが、フィルタリングの意図として残します
ZIMAGE_TARGET_REPLACE_MODULES = ["ZImageTransformerBlock"]

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
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        # 文字列として渡された場合リストに変換
        if isinstance(exclude_patterns, str):
            exclude_patterns = ast.literal_eval(exclude_patterns)

    # exclude if 'norm' in the name of the module
    # 元のスクリプトにあった除外設定を維持
    exclude_patterns.append(r".*(_modulation|_refiner).*")
    
    # kwargsに更新したexclude_patternsを戻す
    kwargs["exclude_patterns"] = exclude_patterns

    # 変更点2: LyCORISのネットワーク作成メソッドを使用
    # 標準のlora.create_networkとは引数の順番が異なるため注意が必要です。
    # LyCORISは (multiplier, dim, alpha, vae, te, unet, ...) の順で受け取ります。
    # また、algo (loha, lokr等) は kwargs 経由で渡されます。
    
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
    # 変更点3: 重みからの読み込みもLyCORIS側を使用
    return lycoris_network.create_network_from_weights(
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference=for_inference,
        **kwargs
    )
