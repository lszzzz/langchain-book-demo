import json
import operator
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain_community.chat_models import ChatTongyi
from langgraph.runtime import get_runtime

from agent_demo.dto.chat_dto import ContextSchema
from agent_demo.workflow.workflow_base import WorkflowBase


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    semantic_search_result: dict


async def intent_recognition(state: AgentState):
    runtime = get_runtime(ContextSchema)

    if not runtime.context["kb_ids"] and not runtime.context["document_ids"]:
        return "direct_llm"

    # 调用llm分析
    prompt = '''
    你是Microcraft，是由Microware科技公司开发和提供的人工智能助手。
    ## 目标
    在确保内容安全合规的情况下通过遵循指令和提供有帮助回复来帮助用户实现他们的目标。

    辨别用户的问题是否可能需要调用内部知识库工具来检索回答：
    如果用户的问题是很常见的问题，例如“你好/你是谁/今天天气如何/”等在大模型知识范围内的问题，请输出 -> 否。否则可能需要调用知识库检索，请输出-> 是
    不要给出多余解释，直接回答 是 或 否
    '''

    system_message = SystemMessage(content=prompt)
    human_message = HumanMessage(content=runtime.context["user_input"])

    model = ChatTongyi(model="qwen-turbo")
    response = await model.ainvoke([system_message, human_message])

    if "是" in response.content:
        return "rag_llm"
    else:
        return "direct_llm"


async def direct_llm(state: AgentState):
    """
    调用 LLM 生成响应
    """
    model = ChatTongyi(model="qwen-turbo")
    response = await model.ainvoke(state["messages"])

    # 将 AI 的回复添加到消息历史
    return {"messages": [response]}


async def rag_llm(state: AgentState):
    rag_result = {
              "value": [
                {
                  "kb_id": "eca487e3-8a52-4875-9862-0324f4cc69b1",
                  "document_id": "6777c9d6-dc33-4abe-b760-d14b97ce3f0f",
                  "text": "- 週期跟踪和戰略制定的有用指標（來源：彭博社）\n- 對股票、債券、房地產和商品等不同資產類別表現的周期性影響\n- 把握市場時機並遵循週期的技術\n- 如何建立投資組合以滿足個人財務需求\n\n\nCourse Duration: 3 hours\n\nMedium of instruction: Cantonese (supplemented with English terminology)\n\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/ IA CPD / MPFA Non-core CPD hours\n\nTrends of Insurance Products Application in Organization\n[保險產品在企業中的應用趨勢]\n(M441)\n\nCourse Outline\n- Big data and information security needs in various kinds of insurance products\n- Impact of HK legal requirement on the needs of director and officers liability insurance\n- Trend of using of keyman insurance and arrangement\n- Global trends of workplace and employee benefits arrangement\n- 各類保險產品的大數據與信息安全需求\n- 香港法律規定對董事及高級職員責任保險需求的影響\n• 關鍵人員保險的使用趨勢和安排\n- 工作場所和員工福利安排的全球趨勢\n\n\n\nCourse Duration: 3 hours\n\nMedium of instruction:\nCantonese (supplemented with English terminology)\n\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/IA CPD / MPFA Non-core CPD hours\n\nBlockchain and Cryptocurrency – An Introduction\n[加密貨幣與區塊鍵入門]\n(M442)\n\nCourse Outline\nWhat is Cryptocurrency?\n- Bitcoin history\n- Cryptocurrency Market\n- Trust or Consensus\n- Technologies behind Bitcoin/cryptocurrency\n- Applications of the technologies\n- Can cryptocurrency replace cash\n- Cryptocurrency Derivatives and ETFs\n- Regulation on Cryptocurrency\n- Cryptocurrency exchange and famous (or infamous) hacks\n- 什麼是加密貨幣？\n- 比特幣歷史\n- 加密貨幣市場\n- 信任或共識\n- 比特幣/加密貨幣背後的技術\n- 技術的應用\n- 加密貨幣可以代替現金嗎\n- 加密貨幣衍生品和 ETF\n- 加密貨幣監管\n- 加密貨幣交易所和著名（或臭名昭著）的黑客\n\n\n\n\nCourse Duration: 3 hours\n\nMedium of instruction: Cantonese (supplemented with English terminology)\n\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/ IA CPD / MPFA Non-core CPD hours\n\nStructure and Operation of Family Office in Mainland China\n[國內家族辦公室的架構與營運]\n(M443)\n\nCourse Outline\n• Definition, function and service capacity of family office\n- Structures and tools commonly used in family offices\n- The daily operation process of the family office\n- 家族辦公室的定義、功能與服務能量\n- 家族辦公室常使用的架構與工具\n- 家族辦公室的日常運作流程梳理\n\n\n\nCourse Duration: 3 hours\n\nMedium of instruction:\nCantonese (supplemented with English terminology)\n\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/ IA CPD / MPFA Non-core CPD hours\nRisk in Financial Markets\n[金融市場的風險]\n(M502)\n\nCourse Outline\nType of financial market risks\n- Riskiness in Risk Measurement\n- Dimension of Diversification\n- Disaster Risk\n金融市場風險類型\n- 風險計量中的風險\n- 多元化維度\n- 灾害風險\n\n\n\nCourse Duration: 3 hours\n\nMedium of instruction:\n\nCantonese (supplemented with English terminology)\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/ IA CPD / MPFA Non-core CPD hours\nAsset Protection in Practice\n[資產保護之實踐]\n(M503)\nCourse Outline\n- Concept of Wealth Planning\n- Overview of Asset Protection\n- Basic Characteristics\n- Asset Protection Strategies\n- 貪富規劃的概念\n資產保護概述\n- 基本特徵\n資產保護策略\n\n\n\nCourse Duration: 2 hours\n\nMedium of instruction:\n\nCantonese (supplemented with\n(English terminology)\n\nCPD recognition:\n2 IFPHK CE credits/ SFC CPT/\nIA CPD / MPFA Non-core CPD\nhours\nMPF - Review and The Way Forward\n[強積金－回顧與前瞻]\n(M508)\n\nCourse Outline\nReview of MPF\n- The current situations and issues\n- How can MPF be improved\n- 檢討強積金\n- 現狀和問題\n• 強積金如何改善\n\n\n\nCourse Duration: 2 hours\n\nMedium of instruction:\nCantonese (supplemented with English terminology)\n\nCPD recognition:\n\n2 IFPHK CE credits/ SFC CPT/IA CPD / MPFA Non-core CPDhours\n\nIFPHK\nAnalyzing Client Needs and Matching Relevant Products\n[分析客戶需求並配置相關理財產品]\n(M513)\n\nCourse Outline\n- Understanding the needs of client from client profiles\n1. Qualitative shortfall and quantitative shortfall\n2.Special concerns\n3.Environmental information\n- Analysis of the features of product available\n- Matching the relevant products with client needs\n- Handling client needs out of your service scope\n- 從客戶檔案中了解客戶的需求\n1. 定性短缺和定量短缺\n2. 特別關注\n3. 環境信息\n可用產品特性分析\n- 將相關產品與客戶需求相匹配\n- 處理超出您服務範圍的客戶需求\n\n\n\nCourse Duration: 2 hours\n\nMedium of instruction:\nCantonese (supplemented with English terminology)\n\nCPD recognition:\n2 IFPHK CE credits/ SFC CPT/IA CPD / MPFA Non-core CPD hours\nNegotiation Skill for Financial Planning Professionals\n[專業財務策劃從業員的談判技能]\n(M515)\n\nCourse Outline\n- Introduction\n1. Nature of negotiation\n2. Why negotiate and when?\n- Preparation\n1. Diagnose Needs\n2. Self-assessment\n3. Assessment of the Other Party\n4. Assessment of the Situation\n- Bargaining techniques for win-win and win-lose negotiation approach\n- Managing concession in bargaining\n- Using questions to overcome barriers\n·介紹\n1. 談判的性質\n2. 為什麼談判以及何時談判？\n準備\n1.診斷需求\n3. 自我評估\n3.對另一方的評估\n4. 情況評估\n- 雙贏和雙贏談判方法的談判技巧\n- 在談判中管理讓步\n- 用問題克服障礙\n\n\n\nCourse Duration: 6 hours\n\nMedium of instruction: Cantonese (supplemented with English terminology)\n\nCPD recognition:\n6 IFPHK CE credits/ SFC CPT/ IA CPD / MPFA Non-core CPD hours\n\nOperational Risk Management for Financial Institutions\n[金融機構運營風險管理]\n(M520)\n\nCourse Outline\n- Nature and impacts of operational risks\nRegulatory requirements for operational risks\n- Best practices in operational risk management\n- Business implications of operational risk management\n- 操作風險的性質和影響\n- 操作風險監管要求\n- 操作風險管理的最佳實踐\n- 操作風險管理的業務影響\n\n\n\nCourse Duration: 3 hours\n\nMedium of instruction:\nCantonese (supplemented with English terminology)\n\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/IA CPD / MPFA Non-core CPD hours\n\nIFPHK\nHow to Relate Economic Data to Investment Planning\n[如何將經濟資料與投資規劃相聯繫]\n(M521)\n\nCourse Outline\n- Economic analysis with the use of market indicators\nStockmarket cycle and investment strategies\n- Allocate assets according to their return potential and volatility nature\n- Optimize financial resources to capture investment opportunities\n- 使用市場指標進行經濟分析\n股市週期和投資策略\n- 根據回報潛力和波動性配置資產\n- 優化財務資源以捕捉投資機會\n\n\n\n\nCourse Duration: 3 hours\n\nMedium of instruction:\nCantonese (supplemented with English terminology)\n\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/ IA CPD / MPFA Non-core CPD hours\nAnalysis on Significance of Mainland China High Net Worth Client's Offshore Insurance Policy\n[國內高淨值客戶境外保單效力分析]\n(M526)\n\nCourse Outline\n- Image description and psychological characteristics of typical domestic customers\n- Analysis of overseas policy effectiveness\n1. Golden bachelor\n2. Successful entrepreneur\n3. Housewife\n- 國內典型客戶形象描述及心理特點\n- 境外保單效力分析\n1. 鉻石王老五\n2. 成功企業家\n3.家庭婦女\n\n\n\nCourse Duration: 3 hours\n\nMedium of instruction:\nCantonese (supplemented with English terminology)\n\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/IA CPD / MPFA Non-core CPD hours\nInvesting for Retirement\n[投資為退休]\n(M527)\n\nCourse Outline\n- Retirement – HK situation\n- Retirement – are you ready?\n- Basic retirement planning skills\n- Investment for retirement – some practical tips\n- 退休-香港情況\n- 退休-你準備好了嗎？\n- 基本的退休計劃技能\n退休投資-一些實用技巧\n\n\n\nCourse Duration: 2 hours\n\nMedium of instruction:\nCantonese (supplemented with English terminology)\n\nCPD recognition:\n2 IFPHK CE credits/ SFC CPT/IA CPD / MPFA Non-core CPD hours\nBig Data for Insurance\n[保險業大數據]\n(M528)\n\n\n(M528)\n\nCourse Outline\nIntroduction\n- Emergence of \"Big Data\"\n- Technicalities of \"Big Data\"\n- Applications of \"Big Data\"\n- Business impacts of \"Big Data\"\n- Future development of \"Big Data\"\n·介紹\n- “大數據”的出現\n- “大數據”的技術性\n- “大數據”的應用\n- “大數據”的商業影響\n- “大數據”的未來發展\n\n\n\nCourse Duration: 3 hours\n\nMedium of instruction:\nCantonese (supplemented with English terminology)\n\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/IA CPD / MPFA Non-core CPD hours\nHow to apply Technical Analysis with different Investment Products\n[如何應用技術分析於不同的投資產品]\n(M529)\n\nCourse Outline\n- Basic knowledge of Technical Analysis (TA)\n- Application of TA on ETF\n- Basic knowledge of ETF\n- Difference between ETF and Mutual Fund\n- How to apply TA on ETF\n- Application of TA on other product (FOREX, commodities)\nCase Studies\n- 技術分析基礎知識 (TA)\n- TA在ETF上的應用\n- ETF基礎知識\n- ETF和共同基金的區別\n- 如何在ETF上申請TA\n- TA在其他產品（外匯、商品）上的應用\n- 質例探究\n\n\n\nCourse Duration: 3 hours\n\nMedium of instruction:\nCantonese (supplemented with English terminology)\n\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/IA CPD / MPFA Non-core CPD hours\nCommon Misunderstanding on Retirement Planning\n[退休規劃的常見誤解]\n(M530)\n\nCourse Outline\n- Retirement = job-free lifestyle after decades of hard working?\n- Retirement is a long-term goal with more rooms for adjustment and correction?\n- Will you run out of financial resources in late retirement life?\n- Should pension be the major source of retirement income?\n- Solution\n- 退休=經過幾十年的努力工作後的無工作生活方式？\n- 退休是一個長期目標，有更多調整和修正的空間？\n- 您會在退休後的生活中耗盡財務資源嗎？\n- 雙老金應該是退休收入的主要來源嗎？\n- 解决方案\n\n\n\nCourse Duration: 3 hours\n\nMedium of instruction:\nCantonese (supplemented with English terminology)\n\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/ IA CPD / MPFA Non-core CPD hours\n\nIFPHK\nSocial Media Marketing for Financial Planning in China\n[利用社交媒體做營銷：中國互聯網理財]\n(M531)\n\n\nCourse Outline\n- Internet Finance in China\n- China's fintech leads the world\n- Monopoly by two giants\n- Application in investment and insurance industry\nSocial Media Platform Marketing\n- 2018 Social Media Trends\n- Top 10 popular social platforms in China\n- Social Commerce Marketing Strategy\nsocial media big data\n- Insights into consumer data\n- Scene-based financial management\n- Robo Financial Advisor\n- 中國互聯網理財\n- 中國金融科技領先全球\n-兩大科網巨頭墜斷\n-投資與保險產業應用\n- 社交媒體平台營銷\n- 2018社交媒體趨勢\n- 中國十大人氣社交平台\n- 社交商務營銷策略\n- 社交媒體大數據\n- 洞察消費者數據\n- 場景化理財\n- 機器人理財顧問\n\n\n\n\nCourse Duration: 3 hours\n\nMedium of instruction: Cantonese (supplemented with English terminology)\n\nCPD recognition:\n3 IFPHK CE credits/ SFC CPT/ IA CPD / MPFA Non-core CPD hours\n\nTax Deductible Voluntary Contributions (TVCs) for MPF Intermediaries\n[可扣稅自願性供款強積金中介人培訓]\n(M532)",
                  "index_node_id": "8744eda7-2898-4e2f-8cad-4099e951fab4"
                }
              ]
            }

    json_str = json.dumps(rag_result, ensure_ascii=False, indent=2)

    state["messages"].append(SystemMessage(content=json_str))

    """
    调用 LLM 生成响应
    """
    model = ChatTongyi(model="qwen-turbo")
    response = await model.ainvoke(state["messages"])

    # 将 AI 的回复添加到消息历史
    return {"messages": [response]}


class RagChatFlow(WorkflowBase):

    def __init__(self):
        super().__init__()

    def compile(self, checkpointer):
        workflow = StateGraph(AgentState)
        workflow.add_node("direct_llm", direct_llm)
        # workflow.add_node("semantic_search", semantic_search)
        workflow.add_node("rag_llm", rag_llm)
        workflow.add_conditional_edges(START, intent_recognition, ["direct_llm", "rag_llm"])
        workflow.add_edge("direct_llm", END)
        # workflow.add_edge("semantic_search", "rag_llm")
        workflow.add_edge("rag_llm", END)
        self.graph = workflow.compile(checkpointer)


rag_chatflow = RagChatFlow()
