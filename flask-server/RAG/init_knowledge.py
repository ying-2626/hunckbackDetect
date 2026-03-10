import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG.rag_advanced import AdvancedRAGService

def init_knowledge_base():
    print("正在初始化坐姿健康知识库...")
    
    rag = AdvancedRAGService(
        use_milvus=False,
        use_bm25=True,
        use_fusion=True
    )
    
    knowledge_items = [
        ("长时间低头会导致颈椎前倾，建议调整显示器高度至视线平齐，保持头部自然直立状态。", "ergonomics"),
        ("每坐1小时应站立活动5分钟，可以缓解腰椎和颈椎压力，促进血液循环。", "health_tips"),
        ("正确的坐姿应该是：背部挺直，肩膀放松，双脚平放在地面，膝盖与髋部同高或略低。", "ergonomics"),
        ("显示器应放在视线正前方，距离眼睛约50-70厘米，屏幕顶部与眼睛平齐或略低。", "ergonomics"),
        ("使用符合人体工程学的椅子，有良好的腰靠支撑，座椅高度可以调节。", "ergonomics"),
        ("经常做颈部拉伸运动：慢慢向左右转动头部，每个方向保持5-10秒，可以缓解颈部肌肉紧张。", "stretching"),
        ("肩部放松运动：双肩向上耸起，然后慢慢放下，重复10次，可以缓解肩部疲劳。", "stretching"),
        ("腰椎保护：站立时双手叉腰，慢慢向后仰，保持5秒，重复5次，可以活动腰椎。", "stretching"),
        ("近视预防：保持正确的用眼距离，每用眼20分钟，看向6米外的物体20秒（20-20-20法则）。", "eye_care"),
        ("工作环境调整：确保桌面高度合适，键盘和鼠标放在肘部自然下垂的位置。", "ergonomics"),
        ("不良坐姿的危害：长期低头会导致颈椎病，驼背会影响肺部功能，身体歪斜可能导致脊柱侧弯。", "health_warning"),
        ("简单的办公室运动：坐在椅子边缘，双脚踩地，双手撑在椅子两侧，慢慢抬起臀部，保持3秒后放下，重复10次。", "stretching"),
        ("深呼吸放松：每小时做几次深呼吸，吸气时腹部鼓起，呼气时腹部收紧，有助于放松身体和缓解压力。", "relaxation"),
        ("午睡建议：如果条件允许，午休时最好平躺或使用U型枕，避免趴在桌子上睡觉压迫颈椎。", "health_tips"),
        ("坐姿检查：定期检查自己的坐姿，可以用手机侧面拍照，看看耳朵、肩膀、髋部是否在一条垂直线上。", "ergonomics"),
        ("脚部支撑：如果脚不能平放在地面，使用脚垫，保持膝盖弯曲约90度。", "ergonomics"),
        ("手腕保护：使用键盘和鼠标时，保持手腕自然伸直，避免过度弯曲或伸展。", "ergonomics"),
        ("定时提醒：使用软件或定时器，每30-60分钟提醒自己调整坐姿或起身活动。", "health_tips"),
        ("腰背肌锻炼：趴在床上，双手放在身体两侧，慢慢抬起头部和胸部，保持3秒后放下，重复10次，可以增强腰背肌力量。", "exercise"),
        ("饮水习惯：保持充足的水分摄入，不仅有益健康，还能促使你定期起身接水活动。", "health_tips")
    ]
    
    for content, category in knowledge_items:
        try:
            rag.add_knowledge(content, category, source="initial_knowledge")
            print(f"已添加: {category} - {content[:50]}...")
        except Exception as e:
            print(f"添加失败: {e}")
    
    print("\n知识库初始化完成！共添加", len(knowledge_items), "条知识。")
    
    return rag

if __name__ == "__main__":
    init_knowledge_base()
