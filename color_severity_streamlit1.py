# color_severity_streamlit_v6_fixed.py

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage import color
import json
import io
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  

st.set_page_config(page_title="색각 캘리브레이터", layout="centered")

# --- LAB <-> RGB 변환 ---
def lab_to_rgb(L, a, b):
    """Convert LAB to RGB with proper clamping"""
    lab = np.array([[[L, a, b]]], dtype=np.float64)
    rgb = color.lab2rgb(lab)
    rgb8 = tuple((np.clip(rgb[0,0], 0, 1) * 255).astype(np.uint8))
    return rgb8

def generate_ishihara_plate(number, axis, deltaE, rng, size=600):
    """
    개선 사항:
    1. Protan/Deutan을 L* 밝기로 구분 (Protan은 어둡게, Deutan은 밝게)
    2. ΔE 범위를 더 좁게 (15~60)
    3. 넓은 색 스펙트럼 유지
    4. 숫자 영역 노이즈 감소로 난이도 조절
    """
    
    # 축별 특성 정의
    if axis == 'protan':
        # 적색맹: 어두운 적-녹 계열, a* 축 차이
        L_base_mean = 50.0  # 어둡게
        base_a_center = rng.uniform(0, 20)
        base_b_center = rng.uniform(5, 15)
        relative_a_diff = deltaE * rng.uniform(0.9, 1.1)
        relative_b_diff = 0
        color_range = "어두운 적-녹"
        
    elif axis == 'deutan':
        # 녹색맹: 밝은 적-녹 계열, a* 축 차이
        L_base_mean = 70.0  # 밝게
        base_a_center = rng.uniform(-10, 10)
        base_b_center = rng.uniform(10, 25)
        relative_a_diff = deltaE * rng.uniform(0.9, 1.1)
        relative_b_diff = 0
        color_range = "밝은 적-녹"
        
    elif axis == 'tritan':
        # 청색맹: 청-황 계열, b* 축 차이
        L_base_mean = 60.0
        base_a_center = rng.uniform(-5, 5)
        base_b_center = rng.uniform(-15, 5)
        relative_a_diff = 0
        relative_b_diff = deltaE * rng.uniform(0.9, 1.1)
        color_range = "청-황"
        
    else:
        L_base_mean = 60.0
        base_a_center = 10
        base_b_center = 15
        relative_a_diff = deltaE * 0.9
        relative_b_diff = 0
        color_range = "기본"

    bg_lab_base = (L_base_mean, base_a_center - relative_a_diff/2, base_b_center - relative_b_diff/2)
    fg_lab_base = (L_base_mean, base_a_center + relative_a_diff/2, base_b_center + relative_b_diff/2)

    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    
    font_size = int(size * 0.65)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    text = str(number)
    bbox = mask_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (size - text_w) // 2 - bbox[0]
    text_y = (size - text_h) // 2 - bbox[1]
    mask_draw.text((text_x, text_y), text, font=font, fill=255)
    
    mask_array = np.array(mask)
    
    center_x, center_y = size // 2, size // 2
    radius = int(size * 0.45)
    
    num_circles = 6000  # 더 많은 원
    circles_to_draw = []
    
    for _ in range(num_circles):
        angle = rng.random() * 2 * math.pi
        r_dist = math.sqrt(rng.random())
        r = r_dist * radius
        x = int(center_x + r * math.cos(angle))
        y = int(center_y + r * math.sin(angle))
     
        # 크기 분포
        rand_val = rng.random()
        if rand_val < 0.6:
            circle_radius = rng.randint(2, 4)
        elif rand_val < 0.9:
            circle_radius = rng.randint(5, 8)
        else:
            circle_radius = rng.randint(9, 12)
        
        if 0 <= x < size and 0 <= y < size:
            in_number = mask_array[y, x] > 128
            
            # 밝기 변화 (더 넓은 범위)
            L_variation = 25.0
            random_L = rng.uniform(L_base_mean - L_variation, L_base_mean + L_variation)
            random_L = np.clip(random_L, 0, 100)
            
            current_fg_lab = (random_L, fg_lab_base[1], fg_lab_base[2])
            current_bg_lab = (random_L, bg_lab_base[1], bg_lab_base[2])
            
            # 숫자 영역은 노이즈 적게 (더 쉽게), 배경은 노이즈 많이 (더 어렵게)
            if in_number:
                contamination = 0.05  # 5% 오염
                final_base_lab = current_fg_lab if rng.random() > contamination else current_bg_lab
                # 숫자 영역 노이즈 감소
                L_noise = rng.uniform(-8, 8)
                a_noise = rng.uniform(-20, 20)
                b_noise = rng.uniform(-20, 20)
            else:
                contamination = 0.08  # 8% 오염
                final_base_lab = current_bg_lab if rng.random() > contamination else current_fg_lab
                # 배경 노이즈 증가
                L_noise = rng.uniform(-18, 18)
                a_noise = rng.uniform(-40, 40)
                b_noise = rng.uniform(-40, 40)
            
            final_lab = (
                np.clip(final_base_lab[0] + L_noise, 0, 100),
                final_base_lab[1] + a_noise,
                final_base_lab[2] + b_noise
            )
            
            final_rgb = lab_to_rgb(*final_lab)
            
            # 약간의 RGB 노이즈
            r_val, g_val, b_val = [int(np.clip(int(c) + rng.randint(-3, 4), 0, 255)) for c in final_rgb]
            circles_to_draw.append((x, y, circle_radius, (r_val, g_val, b_val)))
    
    for x, y, r, col in circles_to_draw:
        bbox = [x-r, y-r, x+r, y+r]
        draw.ellipse(bbox, fill=col)

    return img

class AdaptiveStaircase:
    """
    v6.1 개선 사항:
    - 2-down-1-up 규칙 (v6.0 markdown 기준)
    - 'Reversal(반전)' 기반으로 step_size를 조절 (Coarse-to-Fine)
    """
    def __init__(self, deltas, start_index=None):
        self.deltas = sorted(deltas, reverse=True) # [90, 86, 82, ...]
        self.index = start_index if start_index is not None else len(self.deltas) // 3
        self.history = []
        self.consecutive_correct = 0
        
        self.step_size_large = 3  # 큰 탐색 (3칸)
        self.step_size_small = 1  # 정밀 탐색 (1칸)
        self.step_size = self.step_size_large
        self.reversals = 0         # 방향 전환 횟수
        self.last_direction = None # 마지막 이동 방향
    
    def current_delta(self):
        return self.deltas[self.index]
    
    def record(self, correct):
        self.history.append((self.current_delta(), int(correct)))
        current_direction = None
        
        if correct:
            self.consecutive_correct += 1
            if self.consecutive_correct >= 2:
                self.consecutive_correct = 0
                # 2번 맞힘: 어렵게 (index 증가)
                new_index = min(len(self.deltas) - 1, self.index + self.step_size)
                if new_index != self.index:
                    self.index = new_index
                    current_direction = 'down' # 난이도 하락(down)
        else:
            self.consecutive_correct = 0
            # 1번 틀림: 쉽게 (index 감소)
            new_index = max(0, self.index - self.step_size)
            if new_index != self.index:
                self.index = new_index
                current_direction = 'up' # 난이도 상승(up)

        # 방향 전환(reversal) 감지
        if current_direction and self.last_direction:
            if current_direction != self.last_direction:
                self.reversals += 1
                # 2번째 방향 전환부터 정밀 탐색(step=1)으로 변경
                if self.reversals >= 2:
                    self.step_size = self.step_size_small
                    
        if current_direction:
            self.last_direction = current_direction

# --- Streamlit UI ---
st.title("🎨 개인 맞춤형 색각 캘리브레이터 ")
st.markdown("""
AI 색 보정을 위한 정밀한 개인 색각 프로파일링 도구입니다.

**주의**: 교육·연구 목적이며, 임상 진단용이 아닙니다.
""")

with st.sidebar:
    st.header("⚙️ 설정")
    
    axis_options = {
        "🔴 적색맹 (Protan)": "protan",
        "🟢 녹색맹 (Deutan)": "deutan",
        "🔵 청색맹 (Tritan)": "tritan",
        "🎨 종합 검사 (Mix)": "mix"
    }
    
    selected_option_korean = st.selectbox(
        "측정 유형",
        options=list(axis_options.keys())
    )
    axis = axis_options[selected_option_korean]
    
    n_trials = st.slider('총 문항 수', 20, 40, 30) 
    #seed = st.number_input('시드', value=42)
    
    st.markdown("---")
    st.markdown("### 💡 팁")
    st.info("""
    - 조명을 밝게 하세요
    - 모니터를 정면에서 보세요
    - 천천히 집중해서 보세요
    - 안 보이면 '추측'하지 마세요
    """)
    
    start = st.button('테스트 시작', type='primary')

# 세션 초기화
for key in ['running', 'stair', 'stair_p', 'stair_d', 'stair_t', 'trial', 'axis', 'responses']:
    if key not in st.session_state:
        st.session_state[key] = False if key == 'running' else ([] if key == 'responses' else None)

if start:
    st.session_state.running = True
    st.session_state.trial = 0
    st.session_state.axis = axis
    st.session_state.responses = []
    # 매번 다른 시드 생성 (현재 시각 기반)
    import time
    st.session_state.session_seed = int(time.time() * 1000) % 100000
    
    # 더 좁고 어려운 ΔE 범위
    deltas = list(np.linspace(90, 15, 20))  # 90~15, 20단계
    
    if axis == 'mix':
        st.session_state.stair_p = AdaptiveStaircase(deltas)
        st.session_state.stair_d = AdaptiveStaircase(deltas)
        st.session_state.stair_t = AdaptiveStaircase(deltas)
        
        # 공정한 Mix를 위해 문항 리스트를 미리 생성하고 섞음
        n_per_axis = n_trials // 3
        remainder = n_trials % 3
        trial_list = (['protan'] * n_per_axis) + (['deutan'] * n_per_axis) + (['tritan'] * n_per_axis)
        trial_list += ['protan', 'deutan', 'tritan'][:remainder] # 남은 문항 배분
        
        # 세션 시드를 사용해 이 리스트를 섞음
        np.random.RandomState(st.session_state.session_seed).shuffle(trial_list)
        st.session_state.mix_trial_list = trial_list
        
    else:
        st.session_state.stair = AdaptiveStaircase(deltas)
        
    st.rerun()

if st.session_state.running:
    progress = st.session_state.trial / n_trials
    st.progress(progress)
    st.markdown(f"### 문항 {st.session_state.trial + 1} / {n_trials}")
    
    rng = np.random.RandomState(st.session_state.session_seed + st.session_state.trial)
    num = rng.randint(0, 10)
    current_axis = st.session_state.axis
    
    if current_axis == 'mix':
        # 무작위 선택(rng.choice) 대신, 미리 섞어둔 리스트에서 순서대로 가져옴
        current_trial_axis = st.session_state.mix_trial_list[st.session_state.trial]
        delta = st.session_state[f'stair_{current_trial_axis[0]}'].current_delta()
        st.session_state.current_trial_axis_for_mix = current_trial_axis
    else:
        current_trial_axis = current_axis
        delta = st.session_state.stair.current_delta()
        
    plate = generate_ishihara_plate(num, current_trial_axis, delta, rng)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(plate, use_container_width=True)
        if current_axis == 'mix':
            axis_emoji = {'protan': '🔴', 'deutan': '🟢', 'tritan': '🔵'}
            st.caption(f"{axis_emoji[current_trial_axis]} {current_trial_axis} | ΔE ≈ {delta:.1f}")
        else:
            st.caption(f"색차 ΔE ≈ {delta:.1f}")
    
    with col2:
        st.markdown("**숫자를 찾으세요**")
        st.caption("안 보이면 '패스'를 누르세요")
        answer = st.text_input('숫자 입력 (0-9)', key=f'ans_{st.session_state.trial}', label_visibility="collapsed")
        
        col_submit, col_pass = st.columns(2)
        with col_submit:
            submit = st.button('제출', key=f'sub_{st.session_state.trial}', type='primary', use_container_width=True)
        with col_pass:
            pass_btn = st.button('⏭패스', key=f'pass_{st.session_state.trial}', use_container_width=True)
    
    if submit or pass_btn:
        if pass_btn:
            guess = -1  # 패스 표시
            correct = False
            st.warning(f"패스 (정답: {num})")
        else:
            try:
                guess = int(answer.strip())
                if not 0 <= guess <= 9:
                    st.error("0-9 사이 숫자만 가능합니다")
                    st.stop()
            except:
                st.error("숫자를 입력하세요")
                st.stop()
            
            correct = (guess == num)
            if correct:
                st.success("정답!")
            else:
                st.error(f"오답 (정답: {num})")
        
        # Staircase 업데이트
        if st.session_state.axis == 'mix':
            axis_to_record = st.session_state.current_trial_axis_for_mix[0]
            st.session_state[f'stair_{axis_to_record}'].record(correct)
            recorded_axis = st.session_state.current_trial_axis_for_mix
        else:
            st.session_state.stair.record(correct)
            recorded_axis = st.session_state.axis

        st.session_state.responses.append({
            'trial': st.session_state.trial,
            'number': num,
            'guess': guess,
            'correct': correct,
            'axis': recorded_axis,
            'delta': delta
        })
        
        st.session_state.trial += 1
        if st.session_state.trial >= n_trials:
            st.session_state.running = False
        
        st.rerun()

if not st.session_state.running and st.session_state.responses:
    st.success('✨ 테스트 완료!')

    def calculate_threshold_and_confidence(stair_history):
        """
        Psychometric + reversal 기반 역치 추정 (v7 스타일)
        Returns: (threshold: float or None, confidence: float 0..1)
        stair_history: list of (deltaE, correct) tuples
        """
        # 최소 데이터 검사
        if len(stair_history) < 10:
            return None, 0.0

        # --- 1) 반전점 탐지 (초기 2개 반전 무시) ---
        up_reversals = []    # 'up' -> 'down' (peak)
        down_reversals = []  # 'down' -> 'up' (valley)
        last_direction = None
        consecutive_correct = 0
        reversals_found = 0

        for i in range(len(stair_history)):
            delta, correct = stair_history[i]
            current_dir = None

            if correct:
                consecutive_correct += 1

                if consecutive_correct >= 2:
                    consecutive_correct = 0
                    current_dir = 'down'
            else:
                consecutive_correct = 0
                current_dir = 'up'

            if current_dir and last_direction and current_dir != last_direction:
                reversals_found += 1
                if reversals_found > 2:
                    if current_dir == 'up':
                        down_reversals.append(delta)
                    else:
                        up_reversals.append(delta)

            if current_dir:
                last_direction = current_dir

        # --- 2) 반전 평균 기반 역치 후보 ---
        if up_reversals and down_reversals:
            reversal_mean = (np.mean(up_reversals) + np.mean(down_reversals)) / 2.0
        elif up_reversals:
            reversal_mean = float(np.mean(up_reversals))
        elif down_reversals:
            reversal_mean = float(np.mean(down_reversals))
        else:
            # 반전이 충분히 일어나지 않음 (데이터 부족 또는 한쪽으로 쏠림)
            # 이 경우 피팅에 의존해야 함
            reversal_mean = np.mean([d for d, _ in stair_history[-5:]]) # 임시방편
            if not reversal_mean: return None, 0.0

        # --- 3) Psychometric fitting (로지스틱) ---
        def psychometric_func(deltaE, alpha, beta):
            # 0.5 (chance) to 1.0 (perfect)
            # 2-down-1-up은 ~70.7% 지점을 찾습니다.
            return 1.0 / (1.0 + np.exp(-(deltaE - alpha) / beta))

        deltaE_arr = np.array([d for d, _ in stair_history])
        corrects = np.array([int(c) for _, c in stair_history])

        try:
            popt, _ = curve_fit(
                psychometric_func,
                deltaE_arr,
                corrects,
                p0=[np.mean(deltaE_arr), 5.0],
                bounds=([0.0, 0.1], [100.0, 20.0]),
                maxfev=5000
            )
            alpha, beta = popt
    
            threshold_model = float(alpha + 0.881 * beta) # 70.7% 지점

            # model fit confidence (R^2-like)
            y_pred = psychometric_func(deltaE_arr, alpha, beta)
            ss_res = np.sum((corrects - y_pred) ** 2)
            ss_tot = np.sum((corrects - np.mean(corrects)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            confidence = float(np.clip(r2, 0.0, 1.0))

        except Exception:
            # 피팅 실패 시 반전 평균을 사용하고 낮은 신뢰도 리턴
            threshold_model = float(reversal_mean)
            confidence = 0.25

        # --- 4) 결합 (가중 결합: 모델 우선, 반전 보정) ---
        final_threshold = float((threshold_model * 0.7) + (reversal_mean * 0.3))

        # --- 5) 안정성 보정 (데이터 부족 시 confidence 축소) ---
        if confidence < 0.4 and len(stair_history) < 20:
            confidence *= (len(stair_history) / 20.0)
            
        # 75% 지점이 아닌 70.7% 지점을 찾도록 수정
        # (로지스틱 함수가 0.5가 아닌 0에서 1로 피팅되므로 75% -> 70.7% 변경)
        # 75% -> solve(p=0.75) -> x = a + b * log(3)
        # 70.7% -> solve(p=0.707) -> x = a + b * log(0.707/(1-0.707)) = a + b * log(2.41) = a + b * 0.88
        
        # (재검토) 로지스틱 함수를 0~1로 피팅했으므로 75%가 아니라
        # 2-down-1-up의 목표점인 70.7% 지점을 찾는 것이 맞다.
        # (위 코드에서 threshold_model 계산을 70.7% 지점으로 수정함)

        return round(final_threshold, 2), round(float(np.clip(confidence, 0.0, 1.0)), 2)

    def interpret_threshold(thresh, axis_type):
        """역치를 사람이 읽기 쉬운 문장으로 변환 (단순화)"""
        if thresh is None:
            return "측정 불가", "데이터 부족"

        color_name = {
            'protan': '빨간색',
            'deutan': '초록색',
            'tritan': '파란색/노란색'
        }

        if thresh < 20:
            level = "매우 우수"
        elif thresh < 30:
            level = "우수"
        elif thresh < 40:
            level = "보통"
        elif thresh < 50:
            level = "약간 어려움"
        else:
            level = "어려움"

        desc = f"{color_name.get(axis_type, '색상')} 구분에 {level} 수준입니다."

        return level, desc

    total_correct = sum([r['correct'] for r in st.session_state.responses])
    total_trials = len(st.session_state.responses)
    accuracy = (total_correct / total_trials) * 100 if total_trials > 0 else 0

    st.header("측정 결과")
    st.warning("이 결과는 의학적 진단이 아닙니다. 교육/연구용 참고 자료입니다.")
    
    result_data = {}

    if st.session_state.axis == 'mix':
        st.subheader("🎨 종합 검사 결과")
        
        thresholds = {}
        confidences = {}
        thresholds['protan'], confidences['protan'] = calculate_threshold_and_confidence(st.session_state.stair_p.history)
        thresholds['deutan'], confidences['deutan'] = calculate_threshold_and_confidence(st.session_state.stair_d.history)
        thresholds['tritan'], confidences['tritan'] = calculate_threshold_and_confidence(st.session_state.stair_t.history)
        
        c1, c2, c3 = st.columns(3)
        for col, (name, axis_key, emoji) in zip(
            [c1, c2, c3],
            [("적색맹", "protan", "🔴"), ("녹색맹", "deutan", "🟢"), ("청색맹", "tritan", "🔵")]
        ):
            thresh = thresholds[axis_key]
            conf = confidences[axis_key]
            level, desc = interpret_threshold(thresh, axis_key)
            
            col.metric(
                f"{emoji} {name} (신뢰도: {conf*100:.0f}%)", 
                f"{thresh:.1f}" if thresh else "N/A",
                delta=level if thresh else None,
                help=f"{desc} (신뢰도 {conf*100:.0f}%)"
            )
        
        st.markdown("### 상세 해석")
        for axis_key, name, emoji in [("protan", "적색맹", "🔴"), ("deutan", "녹색맹", "🟢"), ("tritan", "청색맹", "🔵")]:
            thresh = thresholds[axis_key]
            conf = confidences[axis_key]
            level, desc = interpret_threshold(thresh, axis_key)
            
            if thresh:
                with st.expander(f"{emoji} {name} - {level} (ΔE {thresh:.1f} / 신뢰도 {conf*100:.0f}%)"):
                    st.write(desc)
                    if conf < 0.6:
                        st.warning(f"신뢰도가 낮습니다. ({conf*100:.0f}%). 문항 수를 늘려 재측정하는 것을 권장합니다.")
                    st.caption(f"측정된 최소 색차: {thresh:.1f} ΔE (낮을수록 우수)")
        
        result_data = {
            'type': 'mix',
            'thresholds': thresholds,
            'confidences': confidences,
            'accuracy': accuracy,
            'responses': st.session_state.responses
        }

    else:
        # --- 단일 모드 ---
        thresh, conf = calculate_threshold_and_confidence(st.session_state.stair.history)
        level, desc = interpret_threshold(thresh, st.session_state.axis)
        
        c1, c2 = st.columns(2)
        c1.metric(
            f"색각 역치 (신뢰도: {conf*100:.0f}%)",
            f"{thresh:.1f}" if thresh else "N/A",
            delta=level if thresh else None,
            help=f"{desc} (신뢰도 {conf*100:.0f}%)"
        )
        c2.metric("정답률", f"{accuracy:.1f}%")

        st.markdown("### 🔍 해석")
        st.info(desc)
        
        if thresh:
            if conf < 0.6:
                st.warning(f"측정 신뢰도가 낮습니다. ({conf*100:.0f}%). 테스트가 역치에 수렴하지 못했을 수 있습니다. 문항 수를 늘려 재측정해 보세요.")

            if thresh > 45:
                st.warning("""
                높은 역치가 측정되었습니다.
                
                **가능한 원인:**
                - 모니터 밝기/대비가 낮음
                - 주변 조명이 어두움
                - 실제 색각 민감도 차이
                
                **권장 사항:**
                1. 모니터 설정을 조정하고 재측정
                2. 밝은 곳에서 재측정
                3. 지속적으로 높게 나오고 일상생활에서 불편함이 있다면 전문의 상담
                """)
        
        result_data = {
            'type': st.session_state.axis,
            'threshold': thresh,
            'confidence': conf,
            'level': level,
            'description': desc,
            'accuracy': accuracy,
            'responses': st.session_state.responses
        }

    # 그래프
    st.markdown("### 측정 과정")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if st.session_state.axis == 'mix':
        colors_map = {'protan': 'red', 'deutan': 'green', 'tritan': 'blue'}
        for axis_type, stair_obj, name in [
            ('protan', st.session_state.stair_p, '🔴 Protan'),
            ('deutan', st.session_state.stair_d, '🟢 Deutan'),
            ('tritan', st.session_state.stair_t, '🔵 Tritan')
        ]:
            if stair_obj and stair_obj.history: # stair_obj가 None이 아닌지 확인
                deltas = [e[0] for e in stair_obj.history]
                trials = range(1, len(deltas) + 1)
                ax.plot(trials, deltas, 'o-', label=name, alpha=0.7, color=colors_map[axis_type])
        ax.set_title('종합 검사 - 유형별 진행')
    else:
        if st.session_state.stair and st.session_state.stair.history: # stair_obj가 None이 아닌지 확인
            hist = st.session_state.stair.history
            deltas = [e[0] for e in hist]
            corrects = [e[1] for e in hist]
            trials = range(1, len(deltas) + 1)
            
            colors = ['green' if c else 'red' for c in corrects]
            ax.scatter(trials, deltas, c=colors, s=100, alpha=0.6, label='문항 (초록=정답, 빨강=오답)')
            ax.plot(trials, deltas, 'k--', alpha=0.3)
            
            thresh, conf = calculate_threshold_and_confidence(hist)
            if thresh:
                ax.axhline(thresh, color='blue', linestyle=':', linewidth=2, label=f'측정 역치 ≈ {thresh:.1f} (신뢰도 {conf*100:.0f}%)')
            
            ax.set_title(f'{st.session_state.axis} 축 측정 과정')
    
    ax.set_xlabel('문항 번호')
    ax.set_ylabel('색차 ΔE')
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # 데이터 다운로드
    with st.expander("데이터 다운로드"):
        st.json(result_data)
        
        buf = io.BytesIO()
        buf.write(json.dumps(result_data, ensure_ascii=False, indent=2, default=str).encode())
        buf.seek(0)
        st.download_button('JSON 다운로드', buf, f'color_test_{st.session_state.axis}.json', 'application/json')

elif not st.session_state.running:
    st.info("사이드바에서 설정 후 '테스트 시작'을 누르세요")
    
    st.markdown("### 예시 이미지")
    st.caption("실제 테스트에서는 훨씬 어려운 문제가 나옵니다!")
    
    cols = st.columns(3)
    
    example_rng = np.random.RandomState(42) 
    
    with cols[0]:
        ex = generate_ishihara_plate(5, 'protan', 35, example_rng, 300)
        st.image(ex, caption="🔴 Protan (어두운 적-녹)")
    with cols[1]:
        ex = generate_ishihara_plate(7, 'deutan', 35, example_rng, 300)
        st.image(ex, caption="🟢 Deutan (밝은 적-녹)")
    with cols[2]:
        ex = generate_ishihara_plate(2, 'tritan', 35, example_rng, 300)
        st.image(ex, caption="🔵 Tritan (청-황)")

st.markdown("---")
st.caption("⚠️ 본 도구는 교육·연구용이며, 임상 진단 목적으로 사용할 수 없습니다.")