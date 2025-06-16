我现在正在进行一个项目，用来训练一个深度学习模型，你需要按照我的要求帮我构建合理的项目目录，编写相关脚本，不需要具体的运行脚本测试，具体要求如下：
需要使用conda创建一个虚拟环境，命名为ar_pred，python版本为3.10，后续的相关requirements均安装在这个虚拟环境下；
使用git进行版本管理，并创建合适的.gitignore；
完成之后，推送到远程仓库，我已创建相关仓库，你只需要git push origin main即可；
训练的最优、结束checkpoints和中间checkpoints保存在checkpoints文件夹下；
训练时记录Log，保存在log文件夹下;
训练产生的临时文件保存在tmp文件夹下；
wandb产生的数据保存在wandb文件夹下；
训练使用的dataloader和其他工具文件存放在util文件夹下；
训练采用yaml配置文件来进行配置，相关的配置文件存放在config文件夹下；
训练使用的model存放在models文件夹下；
训练脚本命名为train_ar_pred_video_vanilla.py，放在根目录下，需要读取config中的配置文件,使用wandb记录训练数据，保存Log，使用tqdm展示训练进度;

训练读取的数据来自配置文件中的一个路径，比如/sharefiles4/qubohuan/；他们分别存在这个路径的不同的文件夹下，文件夹包括：r_npy（13个气压层）,t_npy（13个气压层）,t2m_npy（单个气压层）,tp_npy（单个气压层）,u_npy（13个气压层）,u10_npy（单个气压层）,v_npy（13个气压层）,v10_npy（单个气压层）,z_npy（13个气压层）;对于单个气压层的变量，例如t2m_npy,他们的文件夹结构如下：
t2m_npy
├── 1979
├── 1980
├── 1981
├── 1982
├── 1983
├── 1984
├── 1985
├── 1986
├── 1987
├── 1988
├── 1989
├── 1990
├── 1991
├── 1992
├── 1993
├── 1994
├── 1995
├── 1996
├── 1997
├── 1998
├── 1999
├── 2000
├── 2001
├── 2002
├── 2003
├── 2004
├── 2005
├── 2006
├── 2007
├── 2008
├── 2009
├── 2010
├── 2011
├── 2012
├── 2013
├── 2014
├── 2015
├── 2016
├── 2017
├── 2018
├── mean_std.json
└── min_max.json；
mean_std.json存放均值和方差，不过现存这些数据已经是标准化之后的；
mean_std.json内容示例如下：
{
    "mean": [
        278.45648193359375
    ],
    "std": [
        21.26502799987793
    ]
};
对于13个气压层的变量，他们的文件夹结构示例如下：
r_npy
├── level_00
├── level_01
├── level_02
├── level_03
├── level_04
├── level_05
├── level_06
├── level_07
├── level_08
├── level_09
├── level_10
├── level_11
└── level_12
每一个level_xx之下都有着和单层变量类似的文件夹结构，也有mean_std.json；
对于每个具体的年份（1979-2018）的文件夹下，存放着该变量该气压层某年的逐小时数据，例如/datasets/t2m_npy/1979下：
.
├── 1979-0000.npy
├── 1979-0001.npy
...
└── 1979-8759.npy
对于非闰年，应该是8760个npy数据；对于闰年，为8784个数据；每一个npy都代表着某个变量在某个气压层在某年的某个时间节点的数据；
你需要读取每6小时（或者逐小时，让我在配置文件中更改）的npy数据；
这样的话，我们共计有69个气压层（5个13层变量，4个单层变量）的变量的数据，简称69个变量；对于每个npy文件，它们具有同样的形状和类型，为(128,256),类型: float32;
我们最终会使用这69个变量中的k个，需要你在配置文件中让我能够更改；我们最少使用9个变量，即单层变量直接使用，多层变量选择“level_00”；
我们将k个变量在同一时间节点的数据拼合在一起，变成(128,256,k)，将其视作一个样本（也可以视作一帧图像，只不过传统图像是RGB三个通道，我们是k个通道）；
假设我们使用1979年到2015年的数据，使用每六小时数据（每小时的数据在后续计算中同理），也就是每天4个时间节点，总共54056个样本；
每个数据集划分为：1979-2013年作为训练集也就是第（1，51136）个样本， 2014年用作测试集也就是第（51137，52596）个样本 ，2015年用于验证也就是第（52597，54056）个样本；
每个样本代表着一个时间点，我们需要进行的是一个预测（重建）任务，即通过过去若干个时刻的样本，去预测未来某个时刻的时间点的样本，也就是说，假设我的时间窗口设为8，在测试和推理时，我输入1-7时刻的原始数据，模型需要输出第8个时刻的预测样本，然后与第8时刻的原始样本进行loss计算；不需要滚动计算，也就是每次我都会提供真实的原始数据去预测，不需要使用预测的数据去预测;
dataloader加载数据时需要检查一下数据的完整性，如果缺少了某个变量某个时间点的数据，用log提示我，并且使用距离它最近的两个时间点的数据插值作为替代；
样本进入模型前，需要先使用其对应的mean_std.json中的均值和方差进行标准化；

随机初始化使用的seed设为42；

模型的接口要封装在models/下不同名的具体模型文件中，使我能够做到通过修改config中的配置，来使用models下的不同模型进行预测（重构）任务；

我的第一个模型命名为ar_pred_video_vanilla.py，它的具体架构和流程如下(以下关于维度的描述中自动忽略了batchsize，你写的时候记得补上)：

我们的这个任务其实可以是视作一个自回归的任务，假设我们输入的时间窗口为t，t默认=8，那么我们的目的就是，将这t个时刻的样本送入Transformer模型中学习他们的上下文和彼此的联系，从而达到能够预测某一时刻的样本的目的。
具体来说，每个batch我们输入的是t个连续时刻的(128,256,k)的样本;我们首先将其在变量维度上叠加，使其变成(128,256,k*t)；然后我们使用ViT的patch embeddings技术,将这个张量token化,每个token的大小为(16,16,k)（这个token大小的设置需要我能在配置文件中更改），这样我们总计有：（128/16）*（256/16）*（k*t/k）=8*16*t=128*t个patch,每个patch的大小为16*16*k=256*k，这样我们得到了一个（128*t，16*16*k）的向量；我们首先将这个向量通过Embedding Layer (嵌入层)，映射到768维度，这样我们就得到了一个（128*t，768）维度的向量；
然后我们需要对这个(128*t，768)的视频张量进行时空区块划分操作，也就是将Token划分成不同的时空簇，我们默认将这个向量划分成t个group，也就是每128个token视作一个group，划分的规则来划分(划分规则我需要可以在config文件中更改)，总计划分成t个group，我们将每个group视作一个区块；
重点来了，我们需要针对每个区块进行位置编码，即这128*t个token被赋予了t个不同的位置编码；编码后，我们得到的还是（128*t，768）维度的向量，但他在宏观上属于t个区块；我们在后续的某些计算上需要以区块为单位，每个区块大小为（128，768）；
在那之后，我们需要经过一个random_patchify的过程,这个random_patchify过程需要在config中可以设置开关，默认为开；random_patchify=True的时候，向量按区块为单位，随机裁去一个区块作为缺失块，记录其是原有区块中的哪一块（其实我们需要保存得是缺失块的对应的原始信息，由于我们按t划分了区块，那么缺失块必定对应着一个原始的(128,256,k)的样本，（不是Embedding Layer后的），以便在后续过程中计算Loss，我的叙述顺序可能有问题，但请你写代码时保证正确的顺序），以在后续过程中使用其Positional Embedding来引导decoder生成；random_patchify=False的时候，直接裁去最后一块作为缺失块；
裁去区块后，我们得到(128*（t-1）,768)的变量，我们可以将其输入到AR_pred_Transformer模型进行训练；
这个AR_pred_Transformer模型由m个Encoder和1个Decoder组成，默认m的数量为8；
Encoder:具体来说，每个Encoder与Vit的Encoder类似，按顺序由Norm层、带有h个头的Multihead-Attention层,h默认为12，Norm层、Mlp层组成，第一次Norm前与Multihead-Attention层后具有跳跃连接，第二次Norm前与MLP后具有跳跃连接；
此外需要注意的是，每个Multihead-Attention层采用的都是一个逐区块的带自回归掩码的Attention过程，带自回归掩码的Attention过程是指，我们设定一个超参数掩码率“mask_radio”（默认为80%），在进行自注意力计算的时候，我们以区块为单位，先来的区块看不到后来的区块（按掩码率），也就是在计算q,k的点积相似度后，将掩码（-1e9）按掩码率随机加到相似度矩阵上，使得后续的对应Softmax的输出值为0，从而达到一个自回归生成的目的，具体来说，我们第一个区块在计算时只能看到自己，第二个区块在计算时可以看到自己和第一个区块，以此类推；
具体的算法过程你可以参考以下代码，需要注意里面的有些部分是不必要的，比如位置旋转位置编码等，此外还要适配多头注意力：


class SelfAttention(nn.Module):
    """多头自注意力机制，带有旋转位置编码"""
    def __init__(self, dim, num_heads=8, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.attn_dropout = attn_dropout
        self.proj_dropout = nn.Dropout(proj_dropout)
        
        # 旋转位置编码
        self.rope = RoPE(self.head_dim // 2)
        
        # 注意力缓存用于自回归生成
        self.k_cache = None
        self.v_cache = None
    
    def forward(self, x, attn_mask=None, update_cache=True):
        """
        x: [batch, seq_len, dim]
        attn_mask: [batch, 1, seq_len, seq_len] 或 None
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算QKV
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3), qkv)
        
        # 应用旋转位置编码
        q_half, k_half = q[..., :self.head_dim // 2], k[..., :self.head_dim // 2]
        q_half = self.rope(q_half)
        k_half = self.rope(k_half)
        q = torch.cat([q_half, q[..., self.head_dim // 2:]], dim=-1)
        k = torch.cat([k_half, k[..., self.head_dim // 2:]], dim=-1)
        
        # 使用缓存进行自回归生成
        if not self.training and self.k_cache is not None:
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)
        
        # 更新缓存
        if update_cache and not self.training:
            self.k_cache, self.v_cache = k, v
        
        # 注意力计算
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 应用注意力掩码
        if attn_mask is not None:
            attn = attn + attn_mask
        
        # Softmax注意力权重
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        
        # 获取注意力结果
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        out = self.to_out(out)
        out = self.proj_dropout(out)
        
        return out


在经过m个EncoderBlock后，得到（128*(t-1)，768）进入Decoder；decoder的深度为4，宽度为512（均为超参）；
Decoder Block:For the decoder, we take the Transformer decoder with cross attention but without self-attention. 
对于每一次训练，Decoder的Query是随机初始化的，但是初始化后需要加上前文保存的缺失块的Positional Embedding（注意是其Positional Embedding而不是缺失块本身）用于引导序列生成；
在经过交叉注意力的decoder后，得到的向量，我们将其输入一个Head，使其还原至原始样本的形状(128,256,k)，这样我们就得到了一个预测样本，可以使用缺失块对应的的原始样本数据来计算Loss,使用mseloss；然后再进行backward即可完成一步训练流程；

此外，上文中我未提及的部分的act_layer和norm_layer默认使用：
act_layer: nn.Module = nn.GELU,
norm_layer: nn.Module = nn.LayerNorm；
默认使用Adam优化器；
以上是训练过程；
对于推理过程，也就是验证和测试过程，我们输入t-1个样本，去预测t时刻（缺失块，也是时间序列的最后一块）的样本，也就是说，只要提供给模型t-1个样本，和最后的缺失块的Positional Embedding去引导decoder，我们就能得到t时刻的预测样本，从而计算val和test loss；


此外，可能出现在dataloader时的问题就是:样本数可能不能被t整除，这时候默认丢弃时间最靠前的数据直到能被整除；
默认Shuffle=True，Shuffle=True的时候，我们随机从训练集的样本中，抽取t个连续的样本进行训练；Shuffle=False的时候，依次加载不重复的t个样本进行训练，暂时没有滑动窗口机制；
此外，我还需要在训练结束后，可视化一些train、val、test的样本保存在wandb中，方便我观察结果，这就需要你编写一个推理流程；
以上就是train_ar_pred_vanilla.py需要编写的内容和思路，你需要在必要的地方打印并记录log（我希望在模型的每一层都打印一下输入输出变量的形状），有一些我没有考虑到的部分，请你自行补足，并在适当处给出中文注释；
此外，你还需要编写一个inference_ar_pred_video_vanilla.py,用以读取checkpoints，在给定t-1个既定的的样本的情况下，能够产生对应的t时刻的的一个样本；