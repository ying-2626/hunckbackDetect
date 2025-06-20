import requests
import time


class Classifier:
    def __init__(self):
        self.url = "http://172.30.251.69:6666/detect"

    def classify_image(self, image_bytes):
        try:
            start_time = time.time()

            response = requests.post(
                self.url,
                files={"image": ("image.jpg", image_bytes, "image/jpeg")},
                timeout=10
            )

            elapsed = time.time() - start_time
            print(f"远程服务响应时间: {elapsed:.2f}秒, 状态码: {response.status_code}")

            response.raise_for_status()

            result = response.json()
            print(f"远程服务返回结果: {result}")
            return result

        except requests.exceptions.Timeout:
            print("远程服务响应超时")
            return {"error": "远程服务响应超时"}
        except requests.exceptions.RequestException as e:
            print(f"请求远程服务失败: {str(e)}")
            return {"error": f"无法连接远程服务: {str(e)}"}
        except Exception as e:
            print(f"分类处理异常: {str(e)}")
            return {"error": f"处理请求时发生错误: {str(e)}"}
