# ✅ JAX GPU 安装教程（CUDA 11 / 12）

## 1. 卸载旧版本
先清理掉 CPU 版 JAX 以及可能导致冲突的插件：
```bash
pip uninstall -y jax jaxlib jax-cuda11-pjrt jax-cuda12-pjrt jax-cuda11-plugin jax-cuda12-plugin
````

## 2. 确认 CUDA 驱动版本

查看本机 CUDA 版本：

```bash
nvidia-smi
```

* 如果 `CUDA Version: 12.x` → 选择 **CUDA 12 包**
* 如果 `CUDA Version: 11.x` → 选择 **CUDA 11 包**

## 3. 安装 GPU 版 JAX

### CUDA 12 用户

```bash
pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### CUDA 11 用户

```bash
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 4. 验证安装

运行以下代码：

```bash
python -c "import jax; print(jax.devices())"
```

如果成功，会看到类似输出：

```
[JaxLocalDevice(id=0, process_index=0, platform='gpu', ...)]
```

## ⚠️ 注意事项

* 不要再手动安装 `jax-cuda*-pjrt` 或 `jax-cuda*-plugin`，它们已经包含在 `jaxlib` 中。
* 第一次运行 JAX 会有编译开销，第二次运行才会体现 GPU 的高性能。

