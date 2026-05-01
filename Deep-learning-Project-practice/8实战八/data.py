import os
import requests


def download_voc2012(save_dir="."):
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    filename = os.path.join(save_dir, "VOCtrainval_11-May-2012.tar")

    if os.path.exists(filename):
        print(f"文件已存在: {filename}")
        return

    print(f"开始下载 VOC2012 数据集 (约 2GB)...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("下载完成！")

        # 自动解压
        import tarfile
        print("正在解压...")
        with tarfile.open(filename, 'r') as tar:
            tar.extractall(path=save_dir)
        print("解压完成！")

    except Exception as e:
        print(f"下载失败: {e}")
        print("提示：如果官方源无法连接，请检查网络或尝试使用代理。")


# 执行下载
download_voc2012()