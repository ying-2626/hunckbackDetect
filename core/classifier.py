import time
import base64
import os
import json
from openai import OpenAI

def load_config():
    """Load configuration from config.json in the project root."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Traverse up to find config.json
        d = current_dir
        while os.path.dirname(d) != d:
            config_path = os.path.join(d, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            d = os.path.dirname(d)
    except Exception as e:
        print(f"Warning: Failed to load config.json: {e}")
    return {}

class Classifier:
    def __init__(self):
        config = load_config()
        api_key = config.get("MODELSCOPE_API_KEY")
        
        if not api_key:
            raise ValueError("Missing 'MODELSCOPE_API_KEY' in config.json. Please configure it to use Classifier.")

        # 初始化 ModelScope 客户端
        self.client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key=api_key,
        )
        self.model = 'Qwen/Qwen2.5-VL-72B-Instruct' 
    def _encode_image(self, image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')

    def classify_image(self, image_bytes):
        try:
            start_time = time.time()
            
            # 转 Base64
            base64_image = self._encode_image(image_bytes)
            
            # 构造 prompt
            # 要求模型返回 JSON 格式
            prompt = """
请作为一位专业的体态健康专家，分析这张图片中人物的坐姿或体态。
请严格按照以下 JSON 格式输出，不要包含 markdown 标记 (```json ... ```) 或其他多余文本，直接返回 JSON 字符串：
{
    "posture_analysis": "这里填写详细的姿态分析，包括是否有驼背、头前倾、脊柱曲线等",
    "detected_issues": ["具体问题1", "具体问题2"],
    "causes_analysis": "这里分析导致该不良姿态的可能成因（如久坐、显示器高度不当等）",
    "harm_analysis": "这里分析该姿态可能带来的健康危害（如颈椎病、腰肌劳损等）",
    "correction_suggestions": "这里提供具体的矫正建议和改善动作",
    "overall_status": "正常/轻度不良/严重不良"
}
注：detected_issues 字段请列出具体的体态问题关键词，例如 ["驼背", "头前倾", "骨盆前倾", "圆肩", "高低肩"]。如果是正常坐姿，该字段返回空数组 []。
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {
                            'type': 'image_url',
                            'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }],
                stream=False
            )

            elapsed = time.time() - start_time
            content = response.choices[0].message.content
            print(f"ModelScope 响应时间: {elapsed:.2f}秒")
            print(f"分析结果: {content}")

            # 清理可能的 markdown 标记
            import json
            cleaned_content = content.replace("```json", "").replace("```", "").strip()
            
            try:
                analysis_data = json.loads(cleaned_content)
            except json.JSONDecodeError:
                # Fallback if model fails to return valid JSON
                print("Warning: Model output is not valid JSON")
                analysis_data = {
                    "posture_analysis": content,
                    "detected_issues": [],
                    "causes_analysis": "无法解析",
                    "harm_analysis": "无法解析",
                    "correction_suggestions": "无法解析",
                    "overall_status": "未知"
                }

            # 简单的关键词匹配来确定 posture (fallback)
            posture_status = "good"
            status_text = analysis_data.get("overall_status", "正常")
            if "不良" in status_text or "驼背" in status_text:
                posture_status = "bad"

            return {
                'class': posture_status,
                'conf': 0.99,
                'analysis': analysis_data  # 返回结构化数据
            }

        except Exception as e:
            print(f"AI 分析异常: {str(e)}")
            return {"error": f"AI 分析服务暂不可用: {str(e)}"}
