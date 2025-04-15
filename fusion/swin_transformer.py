"""
    2021 ICCV Paper
    PatchMerging模块: 高、宽缩小为原来的一半, 深度翻倍
    W-MSA模块-(MSA模块): 目的是为了减少计算量, 缺点是窗口之间无法进行信息交互
    Shifted Window: 实现不同window之间的信息交互, 使用Masked MSA机制(l层是W-MSA模块, l+1层是Shifted Window模块)
    Relative position bias: Attention(Q,K,V) = SoftMax(QK^t / d**0.5 + B)V
"""


class SwimTransformer(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
