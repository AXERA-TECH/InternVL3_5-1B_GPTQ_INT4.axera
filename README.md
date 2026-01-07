# InternVL3_5-1B_GPTQ_INT4.axera

> InternVL3_5-1B_GPTQ_INT4 DEMO on Axera NPU.

- 目前支持 `Python` 语言, `C++` 代码在开发中.
- 预编译模型可以从 [HuggingFace](https://huggingface.co/AXERA-TECH/InternVL3_5-1B_GPTQ_INT4) 下载.
- 如需自行导出编译 `VIT` 模型请参考 [模型转换](/model_convert/README.md).

## 支持平台

- [x] AX650N
- [ ] AX630C

## Git Clone

首先使用如下命令 `clone` 本项目, 然后进入 `python` 文件夹:

```bash
$ git clone git@github.com:AXERA-TECH/InternVL3_5-1B_GPTQ_INT4.axera.git
$ cd InternVL3_5-1B_GPTQ_INT4.axera/python
```

默认文件夹排布如下:

```bash
(hf) ➜  python git:(main) ✗ tree .
.
├── examples
│   ├── image_0.jpg
│   ├── image_1.jpg
│   ├── image_2.png
│   ├── image_3.png
│   ├── laorenshuaidao.mp4
│   ├── red-panda.mp4
│   └── tuboshu.mp4
├── gradio_demo.py
├── infer_axmodel.py
├── infer_torch.py
├── InternVL3_5-1B_GPTQ_INT4
│   ├── added_tokens.json
│   ├── chat_template.jinja
│   ├── config.json
│   ├── configuration_intern_vit.py
│   ├── configuration_internvl_chat.py
│   ├── conversation.py
│   ├── generation_config.json
│   ├── merges.txt
│   ├── modeling_intern_vit.py
│   ├── modeling_internvl_chat.py
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── processor_config.json
│   ├── README.md
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── video_preprocessor_config.json
│   └── vocab.json
├── InternVL3_5-1B_GPTQ_INT4_axmodel
│   ├── model.embed_tokens.weight.bfloat16.bin
│   ├── model.embed_tokens.weight.float32.bin
│   ├── model.embed_tokens.weight.npy
│   ├── qwen3_p128_l0_together.axmodel
│   ├── qwen3_p128_l10_together.axmodel
│   ├── qwen3_p128_l11_together.axmodel
│   ├── qwen3_p128_l12_together.axmodel
│   ├── qwen3_p128_l13_together.axmodel
│   ├── qwen3_p128_l14_together.axmodel
│   ├── qwen3_p128_l15_together.axmodel
│   ├── qwen3_p128_l16_together.axmodel
│   ├── qwen3_p128_l17_together.axmodel
│   ├── qwen3_p128_l18_together.axmodel
│   ├── qwen3_p128_l19_together.axmodel
│   ├── qwen3_p128_l1_together.axmodel
│   ├── qwen3_p128_l20_together.axmodel
│   ├── qwen3_p128_l21_together.axmodel
│   ├── qwen3_p128_l22_together.axmodel
│   ├── qwen3_p128_l23_together.axmodel
│   ├── qwen3_p128_l24_together.axmodel
│   ├── qwen3_p128_l25_together.axmodel
│   ├── qwen3_p128_l26_together.axmodel
│   ├── qwen3_p128_l27_together.axmodel
│   ├── qwen3_p128_l2_together.axmodel
│   ├── qwen3_p128_l3_together.axmodel
│   ├── qwen3_p128_l4_together.axmodel
│   ├── qwen3_p128_l5_together.axmodel
│   ├── qwen3_p128_l6_together.axmodel
│   ├── qwen3_p128_l7_together.axmodel
│   ├── qwen3_p128_l8_together.axmodel
│   ├── qwen3_p128_l9_together.axmodel
│   └── qwen3_post.axmodel
├── utils
│   └── infer_func.py
└── vit-models
    └── internvl_vit_model_1x3x448x448.axmodel

5 directories, 63 files
```

## 模型转换

关于 `onnx` 和 `axmodel` 的导出、编译参见 [模型转换](./model_convert/README.md) 部分内容.

## 上板部署

- `AX650N` 的设备已预装 `Ubuntu 22.04`
- 以 `root` 权限登陆 `AX650N` 的板卡设备
- 接入互联网, 确保 `AX650N` 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备: `AX650N DEMO Board`、`爱芯派Pro(AX650N)`

### Python API 运行

#### Requirements

```bash
$ mkdir /opt/site-packages
$ cd python
$ pip3 install -r requirements.txt --prefix=/opt/site-packages
``` 

#### 添加环境变量

将以下两行添加到 `/root/.bashrc`(实际添加的路径需要自行检查)后, 重新连接终端或者执行 `source ~/.bashrc`

```bash
$ export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages  
$ export PATH=$PATH:/opt/site-packages/local/bin
``` 

#### 运行

使用 `Gradio API` 交互式对话:

```bash
$ python3 gradio_demo.py --hf_model InternVL3_5-1B_GPTQ_INT4/ --axmodel_path InternVL3_5-1B_GPTQ_INT4_axmodel/ --vit_model vit-models/internvl_vit_model_1x3x448x448.axmodel
```

纯文本对话

![demo_1](assets/demo_1.png)

图像理解

![demo_2](assets/demo_2.png)

---

在 `Axera 开发板` 上运行以下命令开始聊天对话:

```sh
$ cd InternVL3_5-1B_GPTQ_INT4.axera/python
$ python3 infer_axmodel.py --hf_model InternVL3_5-1B_GPTQ_INT4/ --axmodel_path InternVL3_5-1B_GPTQ_INT4_axmodel/ --question "请计算函数[y=2x^2+2]的导数, 并提供 markdown 格式的推理过程"
```

输出结果如下:

```bash
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 5.1-dirty 0fdbfe15-dirty
Model loaded successfully!
slice_indices: [0]
Slice prefill done: 0
answer >> 函数 \( y = 2x^2 + 2 \) 的导数可以通过求导法则来计算。首先，我们对函数中的每一项分别求导：

1. 对于 \( 2x^2 \)，使用幂法则求导：
   \[
   \frac{d}{dx}(2x^2) = 2 \cdot 2x = 4x
   \]

2. 对于常数项 \( 2 \)，其导数为 0，因为常数的导数为 0。

将这两部分的结果相加，得到函数 \( y \) 的导数：
\[
y' = 4x
\]

因此，函数 \( y = 2x^2 + 2 \) 的导数为 \( y' = 4x \)。
```

输入以下命令执行单图像理解任务:

```sh
$ cd InternVL3_5-1B_GPTQ_INT4.axera/python
$ python3 infer_axmodel.py --hf_model InternVL3_5-1B_GPTQ_INT4/ --axmodel_path InternVL3_5-1B_GPTQ_INT4_axmodel/ --question "请描述这幅图" -i examples/image_0.jpg --vit_model vit-models/internvl_vit_model_1x3x448x448.axmodel
```

![image_0.jpg](python/examples/image_0.jpg)

模型推理结果如下:

```bash
[INFO] Model type: 2 (triple core)
[INFO] Compiler version: 5.1-dirty 0fdbfe15-dirty
Model loaded successfully!
slice_indices: [0, 1, 2]
Slice prefill done: 0
Slice prefill done: 1
Slice prefill done: 2
answer >> This image shows a close-up of a red panda with a brown and white face, featuring its distinctive facial markings and ears. The red panda is holding onto a piece of wood or a branch, and it appears to be in a natural or zoo-like setting with green foliage in the background.
```

#### 推理耗时统计

该模型 prefill 阶段存在 9 个可用子图, 共 28 层 Decode Layer, 每个子图耗时如下:

```
g1: 2.208 ms
g2: 2.562 ms
g3: 2.804 ms
g4: 3.173 ms
g5: 3.517 ms
g6: 3.807 ms
g7: 4.079 ms
g8: 4.387 ms
g9: 4.713 ms
```

decode 阶段只有一个子图, 耗时如下:

```
g0: 0.987 ms
```

后处理耗时: 7.954 ms.

- 模型最大 TTFT 为: ~~176.554~~ 31.268 * 28 + 7.954 约为 ~~4951.50~~ 883.458 ms.

- 模型解码速度为: 1000 / (0.987 * 28 + 7.954)  = 28.09 tokens/s.


## 技术讨论

- Github issues
- QQ 群: 139953715
