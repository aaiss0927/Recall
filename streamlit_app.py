import streamlit as st
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_community.callbacks import get_openai_callback
import json

# --- 페이지 설정 ---
st.set_page_config(page_title="기억 회상 봇", page_icon="🧠", layout="wide")
st.title("🧠 기억 회상 봇")

# --- 프롬프트 템플릿 ---
PROMPTS = {
    "generate_question": """당신은 치매 환자가 특정 키워드를 기억해내도록 돕는 질문 생성 AI입니다. 당신의 임무는 주어진 정보를 바탕으로, 환자가 '타겟 키워드'를 스스로 말하도록 유도하는 단 하나의 질문을 만드는 것입니다.

[지시사항]
1. 주어진 '전체 이야기'의 맥락을 완벽히 이해하세요.
2. '타겟 키워드'와 관련된 상황을 떠올리세요.
3. 환자가 '타겟 키워드'를 직접 떠올려 대답할 수 있는, 명확하고 개방적인 질문을 만드세요. (예: "그날 식사는 어디서 하셨나요?", "무엇을 타고 이동하셨나요?", "날씨는 어땠나요?")
4. **절대금지사항**: 생성할 질문에는 아래 '금지 단어 목록'에 있는 단어가 단 하나라도 포함되어서는 안 됩니다. 이것은 가장 중요한 규칙입니다.

[정보]
- 전체 이야기: {story}
- 타겟 키워드: {target_keyword}
- 이미 맞춘 키워드: {recalled_keywords}
- 금지 단어 목록: {forbidden_keywords}

질문을 한 문장으로 생성해주세요.""",

    "analyze_answer": """당신은 사용자의 답변을 분석하는 AI입니다.
    사용자의 답변에 아래 '전체 키워드 목록' 중 어떤 키워드가 포함되어 있는지 분석해주세요.
    답변에 명확히 언급되었거나, 강하게 암시되는 키워드를 모두 찾아 쉼표로 구분된 리스트 형태로만 답변해주세요.
    만약 해당하는 키워드가 없다면 '없음'이라고만 답변해주세요.

    # 전체 키워드 목록:
    {all_keywords}

    # 사용자 답변:
    {answer}""",

    "generate_hint": """당신은 치매 환자에게 힌트를 제공하는 친절한 상담사입니다.
    정답을 알려주지 않으면서, 환자가 막혀있는 키워드를 연상할 수 있도록 도와야 합니다.

    # 전체 이야기:
    {story}

    # 유도하려던 키워드:
    {target_keyword}

    위 키워드를 떠올릴 수 있도록, 더 쉽고 구체적인 힌트 질문을 한 문장으로 생성해주세요."""
}


# --- Session State 초기화 ---
def init_session_state():
    if "phase" not in st.session_state:
        st.session_state.phase = "START"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "story" not in st.session_state:
        st.session_state.story = ""
    if "all_keywords" not in st.session_state:
        st.session_state.all_keywords = []
    if "recalled_keywords" not in st.session_state:
        st.session_state.recalled_keywords = []
    if "remaining_keywords" not in st.session_state:
        st.session_state.remaining_keywords = []
    if "current_keyword" not in st.session_state:
        st.session_state.current_keyword = ""
    if "hint_count" not in st.session_state:
        st.session_state.hint_count = 0
    if "session_tokens" not in st.session_state:
        st.session_state.session_tokens = {"prompt": 0, "completion": 0, "total": 0}

# --- LLM 호출 함수 ---
def call_llm(prompt, parser=None):
    with get_openai_callback() as cb:
        response = llm.invoke(prompt)
        st.session_state.session_tokens["prompt"] += cb.prompt_tokens
        st.session_state.session_tokens["completion"] += cb.completion_tokens
        st.session_state.session_tokens["total"] += cb.total_tokens
    
    if parser:
        return parser.parse(response.content)
    return response.content

# --- 메인 로직 함수 ---
def handle_user_answer():
    if st.session_state.phase == "RECALLING" and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_answer = st.session_state.messages[-1]["content"]
        
        analyze_prompt = PROMPTS["analyze_answer"].format(
            all_keywords=json.dumps(st.session_state.all_keywords, ensure_ascii=False),
            answer=user_answer
        )
        found_keywords_str = call_llm(analyze_prompt)
        
        found_keywords = []
        if found_keywords_str != "없음":
            found_keywords = [k.strip() for k in found_keywords_str.split(',')]
        
        newly_recalled = []
        if st.session_state.current_keyword in found_keywords:
             newly_recalled.append(st.session_state.current_keyword)

        for keyword in found_keywords:
            if keyword in st.session_state.remaining_keywords and keyword not in newly_recalled:
                newly_recalled.append(keyword)
                st.session_state.remaining_keywords.remove(keyword)

        if newly_recalled:
            st.session_state.recalled_keywords.extend(newly_recalled)
            st.session_state.hint_count = 0
        else:
            st.session_state.hint_count += 1
        
        is_current_recalled = st.session_state.current_keyword in st.session_state.recalled_keywords
        if not st.session_state.remaining_keywords and is_current_recalled:
            st.session_state.phase = "COMPLETED"
            final_message = f"""정말 대단하세요! 이야기의 모든 조각들({', '.join(st.session_state.all_keywords)})을 성공적으로 떠올리셨어요.

---

**다시 보는 나의 이야기:**
> {st.session_state.story}

---

이 소중한 기억이 오랫동안 빛나기를 바랍니다. 또 다른 이야기를 나누고 싶으시면 언제든지 들려주세요!"""
            st.session_state.messages.append({"role": "assistant", "content": final_message})
            st.session_state.phase = "DONE"
            st.rerun()
            return

        if st.session_state.hint_count >= 2:
            ai_message = f"괜찮아요, 정답은 '{st.session_state.current_keyword}'이었어요. 다음 기억으로 넘어가 볼까요?"
            st.session_state.recalled_keywords.append(st.session_state.current_keyword)
            
            if st.session_state.remaining_keywords:
                st.session_state.current_keyword = st.session_state.remaining_keywords.pop(0)
                st.session_state.hint_count = 0
            else:
                st.session_state.phase = "COMPLETED"
                st.rerun()
                return
        
        elif st.session_state.hint_count == 1:
            prompt = PROMPTS["generate_hint"].format(story=st.session_state.story, target_keyword=st.session_state.current_keyword)
            ai_message = call_llm(prompt)
        
        else:
            if st.session_state.current_keyword in st.session_state.recalled_keywords:
                if st.session_state.remaining_keywords:
                    st.session_state.current_keyword = st.session_state.remaining_keywords.pop(0)
                else:
                    st.session_state.phase = "COMPLETED"
                    st.rerun()
                    return
            
            # [수정] 금지 단어 목록 생성
            forbidden_list = [st.session_state.current_keyword] + st.session_state.remaining_keywords
            prompt = PROMPTS["generate_question"].format(
                story=st.session_state.story,
                recalled_keywords=json.dumps(st.session_state.recalled_keywords, ensure_ascii=False),
                target_keyword=st.session_state.current_keyword,
                forbidden_keywords=json.dumps(forbidden_list, ensure_ascii=False)
            )
            ai_message = call_llm(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": ai_message})
        st.rerun()

# --- UI 렌더링 ---
try:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    init_session_state()

    # Phase: START - 스토리와 키워드 입력 단계
    if st.session_state.phase == "START":
        st.info("아래 입력창에 회상하고 싶은 이야기와 핵심 기억 조각(키워드)을 입력해주세요.")
        story_input = st.text_area("나의 이야기", height=250, key="story_input")
        keyword_input = st.text_input("기억 조각 (쉼표로 구분하여 입력)", key="keyword_input", placeholder="예: 졸업식, 비, 강당, 상장, 중식")
        
        if st.button("기억 회상 시작하기", type="primary"):
            if story_input and keyword_input:
                st.session_state.story = story_input
                st.session_state.user_keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]
                st.session_state.phase = "INIT"
                st.rerun()
            else:
                st.warning("이야기와 기억 조각을 모두 입력해주세요.")

    # Phase: INIT - 사용자 입력 기반으로 첫 질문 생성
    elif st.session_state.phase == "INIT":
        with st.spinner("기억 회상 준비 중..."):
            st.session_state.all_keywords = st.session_state.user_keywords
            st.session_state.remaining_keywords = st.session_state.all_keywords.copy()
            st.session_state.recalled_keywords = []
            
            if st.session_state.remaining_keywords:
                st.session_state.current_keyword = st.session_state.remaining_keywords.pop(0)
                st.session_state.hint_count = 0

                # [수정] 금지 단어 목록 생성
                forbidden_list = [st.session_state.current_keyword] + st.session_state.remaining_keywords
                q_prompt = PROMPTS["generate_question"].format(
                    story=st.session_state.story,
                    recalled_keywords="[]",
                    target_keyword=st.session_state.current_keyword,
                    forbidden_keywords=json.dumps(forbidden_list, ensure_ascii=False)
                )
                first_question = call_llm(q_prompt)
                st.session_state.messages.append({"role": "assistant", "content": first_question})
                st.session_state.phase = "RECALLING"
                st.rerun()
            else:
                st.error("유효한 기억 조각(키워드)이 없습니다. 다시 입력해주세요.")
                st.session_state.phase = "START"
                st.rerun()
    
    # Phase: RECALLING, COMPLETED, DONE - 대화창 표시 단계
    else:
        col1, col2 = st.columns([2, 1.2])

        with col1:
            st.header("대화창")
            chat_container = st.container(height=600)
            for message in st.session_state.messages:
                with chat_container.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("답변을 입력하세요.", disabled=(st.session_state.phase != "RECALLING")):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
            
            handle_user_answer()

        with col2:
            st.header("기억 회상 상태")
            with st.expander("전체 이야기 보기", expanded=False):
                st.info(st.session_state.story)
            
            st.subheader("✅ 성공한 기억 조각")
            st.markdown(" ".join([f"`{kw}`" for kw in st.session_state.recalled_keywords]) if st.session_state.recalled_keywords else "...")
            
            st.subheader("🧩 남은 기억 조각")
            display_remaining = []
            if st.session_state.current_keyword and st.session_state.current_keyword not in st.session_state.recalled_keywords:
                 display_remaining.append(st.session_state.current_keyword)
            display_remaining.extend(st.session_state.remaining_keywords)

            if display_remaining:
                progress_value = len(st.session_state.recalled_keywords) / len(st.session_state.all_keywords) if st.session_state.all_keywords else 0
                st.progress(progress_value, text=f"{len(st.session_state.recalled_keywords)} / {len(st.session_state.all_keywords)}")
                remaining_tags = " ".join([f"`{kw}`" for kw in display_remaining])
                st.markdown(remaining_tags)
            else:
                st.progress(1.0, "모든 기억 조각을 찾았어요! 🎉")
            
            st.divider()
            st.subheader("비용 추적")
            tokens = st.session_state.session_tokens
            st.metric(label="총 사용 토큰", value=f"{tokens['total']:,}")
            st.caption(f"(입력: {tokens['prompt']:,} / 출력: {tokens['completion']:,})")
            
            if st.button("새로운 대화 시작하기"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

except Exception as e:
    st.error(f"오류가 발생했습니다. OpenAI API 키를 확인해주세요: {e}")
