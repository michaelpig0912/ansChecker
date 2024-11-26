import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import csv

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFont

import tkinter as tk
from tkinter import ttk, Canvas, filedialog, messagebox 
from PIL import Image, ImageTk, ImageDraw, ImageEnhance, ImageFilter
from scipy import ndimage
import easyocr
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import matplotlib.pyplot as plt

# 儲存選擇的區域
regions = []
drawing = False
start_x, start_y = -1, -1

# 在全局範圍初始化 reader（只需要初始化一次）
# reader = easyocr.Reader(['en'])  # 只使用英文識別數字即可

class RegionSelector:
    def __init__(self, image):
        self.root = tk.Toplevel()
        self.root.title("選擇區域")
        
        # 轉換圖片為 PhotoImage
        self.image = Image.fromarray(image)
        self.photo = ImageTk.PhotoImage(self.image)
        
        # 創建畫布
        self.canvas = Canvas(self.root, width=self.image.width, height=self.image.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        
        # 綁定滑鼠事件
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # 建立控制按鈕
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=5)
        
        ttk.Button(self.button_frame, text="完成", command=self.finish).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="清除", command=self.clear).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="復原", command=self.undo).pack(side=tk.LEFT, padx=5)
        
        self.regions = []
        self.current_rect = None
        
    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect = None
        
    def on_drag(self, event):
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="green", width=2
        )
        
    def on_release(self, event):
        if self.current_rect:
            x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
            x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
            self.regions.append((x1, y1, x2, y2))
            self.current_rect = None
            
    def clear(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.regions = []
        
    def undo(self):
        if self.regions:
            self.regions.pop()
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            for region in self.regions:
                self.canvas.create_rectangle(
                    region[0], region[1], region[2], region[3],
                    outline="green", width=2
                )
                
    def finish(self):
        self.root.quit()
        self.root.destroy()

def select_regions(image):
    selector = RegionSelector(image)
    selector.root.mainloop()
    return selector.regions

def convert_pdf_to_images(pdf_path):
    """將 PDF 轉換為圖片列表"""
    try:
        # 開啟 PDF
        pdf = fitz.open(pdf_path)
        images = []
        
        # 處理每一頁
        for page in pdf:
            # 將頁面渲染為圖片，降低解析度以加快處理速度
            pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))  # 改為 150 DPI
            
            # 轉換為 PIL Image
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # 調整圖片大
            screen_width = 1920  # 設定固定值避免創建不必要的 Tk 實例
            screen_height = 1080
            
            width_ratio = (screen_width * 0.8) / image.size[0]
            height_ratio = (screen_height * 0.8) / image.size[1]
            scale = min(width_ratio, height_ratio)
            
            new_width = int(image.size[0] * scale)
            new_height = int(image.size[1] * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            images.append(np.array(image))
        
        pdf.close()
        return images
    except Exception as e:
        print(f"PDF 轉換錯誤: {e}")
        return []

def show_full_page_for_confirmation(image, regions, title="請確認"):
    """使用 tkinter 顯示完整頁面並標記需要確認的區域"""
    root = tk.Toplevel()
    root.title(title)
    
    # 轉換圖片
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    
    # 繪製區域
    for x1, y1, x2, y2 in regions:
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
    
    # 調整圖片大小以適應螢幕
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    scale = min(screen_width / img.width, screen_height / img.height) * 0.9
    new_width = int(img.width * scale)
    new_height = int(img.height * scale)
    
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    
    # 顯示圖片
    label = ttk.Label(root, image=photo)
    label.image = photo  # 保持參考
    label.pack()
    
    return root

def enhance_image(image):
    """替代 cv2 的圖像增強功能"""
    # 轉換為 PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # 增強對比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # 銳化
    image = image.filter(ImageFilter.SHARPEN)
    
    return np.array(image)

def convert_to_grayscale(image):
    """替代 cv2.cvtColor 的灰度轉換"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return np.array(image.convert('L'))

def binary_threshold(gray_image, threshold=128):
    """替代 cv2.threshold 的二值化處理"""
    if isinstance(gray_image, Image.Image):
        gray_image = np.array(gray_image)
    return (gray_image > threshold).astype(np.uint8) * 255

def adaptive_threshold(gray_image, block_size=11, c=2):
    """替代 cv2.adaptiveThreshold 的自適應閾值處理"""
    if isinstance(gray_image, Image.Image):
        gray_image = np.array(gray_image)
    
    # 計算局部平均值
    local_mean = ndimage.uniform_filter(gray_image, size=block_size)
    
    # 應用閾值
    return ((gray_image > local_mean - c) * 255).astype(np.uint8)

def find_contours(binary_image):
    """替代 cv2.findContours 的輪廓檢測"""
    if isinstance(binary_image, Image.Image):
        binary_image = np.array(binary_image)
    
    # 使用 scipy.ndimage 進行連通區域標記
    labeled_array, num_features = ndimage.label(binary_image)
    
    contours = []
    for i in range(1, num_features + 1):
        # 獲取每個連通區域的座標
        coords = np.where(labeled_array == i)
        if len(coords[0]) > 10:  # 過濾太小的區域
            y_coords = coords[0]
            x_coords = coords[1]
            # 獲取邊界框
            contour = {
                'x': min(x_coords),
                'y': min(y_coords),
                'width': max(x_coords) - min(x_coords),
                'height': max(y_coords) - min(y_coords),
                'area': len(coords[0])
            }
            contours.append(contour)
    
    return contours

def get_closest_number(text, confidence):
    """將非數字文本轉換為最相似的數字"""
    number_patterns = {
        '0': ['o', 'O', 'D', '()', 'Q'],
        '1': ['l', 'I', 'i','|>', 'J','j', '|','!', '[', ']'],
        '2': ['z', 'Z', 'S'],
        '3': ['B', 'E'],
        '4': ['A'],
        '5': ['S', 's'],
        '6': ['G', 'b'],
        '7': ['T'],
        '8': ['B'],
        '9': ['g', 'q']
    }
    
    if text.isdigit():
        return text, confidence
        
    for num, patterns in number_patterns.items():
        if text in patterns:
            return num, confidence * 0.8
            
    return '', 0

def is_valid_student_id(text):
    """檢查座號是否有效（只能是數字）"""
    return text.isdigit()

def extract_student_id(image, region):
    """提取座號"""
    try:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        if not region:
            print("警告：未選擇座號區域")
            return "未知"
            
        x1, y1, x2, y2 = region[0]
        
        # 確保圖像是 PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # 儲存原始裁切區域
        roi = image.crop((x1, y1, x2, y2))
        roi.save('debug/student_id_1_original.png')
        
        # 1. 放大圖像
        scale_factor = 2
        roi = roi.resize((int(roi.width * scale_factor), int(roi.height * scale_factor)), Image.Resampling.LANCZOS)
        roi.save('debug/student_id_2_resized.png')
        
        # 2. 增強對比度
        enhancer = ImageEnhance.Contrast(roi)
        roi = enhancer.enhance(2.0)
        roi.save('debug/student_id_3_contrast.png')
        
        # 3. 銳化
        roi = roi.filter(ImageFilter.SHARPEN)
        roi.save('debug/student_id_4_sharp.png')
        
        # OCR 辨識
        result = reader.readtext(
            np.array(roi),
            detail=1,
            paragraph=False,
            decoder='greedy',
            text_threshold=0.2,
            low_text=0.2,
            link_threshold=0.2,
            mag_ratio=2,
            add_margin=0.1
        )
        
        if result:
            # 處理所有辨識結果
            print("\n座號區域的完整辨識結果:")
            processed_results = []
            
            for bbox, text, confidence in result:
                cleaned_text = text.replace('(', '').replace(')', '').replace(' ', '').strip()
                print(f"辨識文字: {cleaned_text} (信心度: {confidence:.2f})")
                processed_results.append((cleaned_text, confidence))
            
            if processed_results:
                processed_results.sort(key=lambda x: x[1], reverse=True)
                best_result = processed_results[0][0]
                best_confidence = processed_results[0][1]
                
                # 檢查結果是否為純數字，以及信心度
                if not is_valid_student_id(best_result) or best_confidence < 0.5:
                    print(f"\n座號需要人工判斷 (辨識結果: {best_result}, 信心度: {best_confidence:.2f})")
                    roi = image.crop((x1, y1, x2, y2))
                    fig = show_image(roi, "座號區域")
                    manual_input = input("請輸入正確的座號: ")
                    plt.close(fig)
                    return manual_input
                    
                print(f"最終選擇的座號: {best_result}")
                return best_result
        
        # 如果無法識別，顯示圖片並請求手動輸入
        print("\n無法自動識別座號，請查看圖片並手動輸入")
        roi = image.crop((x1, y1, x2, y2))
        fig = show_image(roi, "座號區域")
        manual_input = input("請輸入正確的座號: ")
        plt.close(fig)
        return manual_input
            
    except Exception as e:
        print(f"座號辨識錯誤: {str(e)}")
        roi = image.crop((x1, y1, x2, y2))
        fig = show_image(roi, "座號區域")
        manual_input = input("請輸入正確的座號: ")
        plt.close(fig)
        return manual_input

def normalize_text(text):
    """將特殊字符轉換為標準答案格式"""
    # 定義字符轉換規則
    char_map = {
        'A': ['4', '∆', '△', 'Δ'],
        'B': ['8', '13', 'β', 'β'],
        'C': ['(', '〔', '［', '<'],
        'D': ['0', 'O', 'o', 'Q'],
        # 可以繼續添加其他規則
    }
    
    # 清理文字
    cleaned_text = text.replace('(', '').replace(')', '').replace(' ', '').strip().upper()
    
    # 檢查是否需要轉換
    for answer, chars in char_map.items():
        if cleaned_text in chars:
            return answer
    
    return cleaned_text

def show_image(image, title="Image"):
    """顯示圖片的輔助函數"""
    # 創建更小的圖片視窗並設置位置
    fig = plt.figure(figsize=(4, 3))  # 縮小圖片尺寸
    mngr = plt.get_current_fig_manager()
    # 設置窗口位置在左上角
    try:
        mngr.window.wm_geometry("+0+0")  # 對於 TkAgg backend
    except:
        try:
            mngr.window.setGeometry(0, 0, 400, 300)  # 對於 Qt backend
        except:
            pass  # 如果都失敗，就使用默認位置
    
    if isinstance(image, np.ndarray):
        plt.imshow(image)
    else:
        plt.imshow(np.array(image))
    plt.title(title)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.1)
    return fig

def is_valid_answer(text):
    """檢查答案是否有效"""
    if not text:  # 空白是有效的
        return True
    if text.upper() == 'X':  # X 是有效的
        return True
    # 修改為檢查 ABCD
    if text.upper() in ['A', 'B', 'C', 'D']:
        return True
    return False

def perform_ocr(reader, image, stage_name):
    """在每個階段進行 OCR 辨識"""
    try:
        result = reader.readtext(
            np.array(image),
            detail=1,
            paragraph=False,
            decoder='greedy',
            text_threshold=0.2,
            low_text=0.2,
            link_threshold=0.2,
            mag_ratio=2,
            add_margin=0.1
        )
        
        if result:
            text = result[0][1]
            confidence = result[0][2]
            normalized_text = normalize_text(text)
            
            # 檢查信心度和答案有效性
            if confidence < 0.5 or not is_valid_answer(normalized_text):
                return None, confidence
                
            if normalized_text != text:
                print(f"  字符轉換: {text} -> {normalized_text}")
                
            return normalized_text, confidence
        return None, 0
        
    except Exception as e:
        print(f"{stage_name} 階段辨識錯誤: {str(e)}")
        return None, 0

def extract_answers(image, regions, question_numbers):
    """使用純 PIL 和 numpy 的版本"""
    try:
        # 修改為同時支援英文和數字的識別
        reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    except Exception as e:
        print(f"EasyOCR 初始化錯誤: {e}")
        return [''] * len(regions)
    
    results = []
    
    for (x1, y1, x2, y2), q_num in zip(regions, question_numbers):
        try:
            print(f"\n處理題號 {q_num}:")
            stage_results = []
            
            # 確保圖像是 PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # 1. 原始裁切區域
            roi = image.crop((x1, y1, x2, y2))
            roi.save(f'debug/q{q_num}_1_original.png')
            text, conf = perform_ocr(reader, roi, "原始圖像")
            if text:
                print(f"階段1 - 原始圖像: {text} (信心度: {conf:.2f})")
                stage_results.append((text, conf, "原始圖像"))
            
            # 2. 放大圖像
            scale_factor = 2
            roi_resized = roi.resize((int(roi.width * scale_factor), int(roi.height * scale_factor)), Image.Resampling.LANCZOS)
            roi_resized.save(f'debug/q{q_num}_2_resized.png')
            text, conf = perform_ocr(reader, roi_resized, "放大圖像")
            if text:
                print(f"階段2 - 放大圖像: {text} (信心度: {conf:.2f})")
                stage_results.append((text, conf, "放大圖像"))
            
            # 3. 增強對比度
            enhancer = ImageEnhance.Contrast(roi_resized)
            roi_contrast = enhancer.enhance(2.0)
            roi_contrast.save(f'debug/q{q_num}_3_contrast.png')
            text, conf = perform_ocr(reader, roi_contrast, "對比度增強")
            if text:
                print(f"階段3 - 對比度增強: {text} (信心度: {conf:.2f})")
                stage_results.append((text, conf, "對比度增強"))
            
            # 4. 銳化
            roi_sharp = roi_contrast.filter(ImageFilter.SHARPEN)
            roi_sharp.save(f'debug/q{q_num}_4_sharp.png')
            text, conf = perform_ocr(reader, roi_sharp, "銳化")
            if text:
                print(f"階段4 - 銳化: {text} (信心度: {conf:.2f})")
                stage_results.append((text, conf, "銳化"))
            
            # 分析結果
            if stage_results:
                # 按信心度排序
                stage_results.sort(key=lambda x: x[1], reverse=True)
                best_result = stage_results[0]
                final_text = best_result[0]
                confidence = best_result[1]
                
                # 檢查結果是否需要人工判斷
                if confidence < 0.5 or not is_valid_answer(final_text):
                    print(f"\n題號 {q_num} 需要人工判斷 (信心度: {confidence:.2f}, 辨識結果: {final_text})")
                    roi = image.crop((x1, y1, x2, y2))
                    fig = show_image(roi, f"題號 {q_num} 區域")
                    final_text = input(f"請輸入題號 {q_num} 的答案: ")
                    plt.close(fig)
                
                results.append(final_text)
            else:
                print(f"題號 {q_num} 所有階段都無法識別")
                roi = image.crop((x1, y1, x2, y2))
                fig = show_image(roi, f"題號 {q_num} 區域")
                manual_input = input(f"請輸入題號 {q_num} 的答案: ")
                plt.close(fig)
                results.append(manual_input)
                
        except Exception as e:
            print(f"處理題號 {q_num} 時發生錯誤: {str(e)}")
            roi = image.crop((x1, y1, x2, y2))
            fig = show_image(roi, f"題號 {q_num} 區域")
            manual_input = input(f"請輸入題號 {q_num} 的答案: ")
            plt.close(fig)
            results.append(manual_input)
            
    return results

def process_pdf(pdf_path):
    print("正在初始化 OCR 引擎...")
    try:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    except Exception as e:
        print(f"EasyOCR 初始化錯誤: {e}")
        return None
        
    print("正在轉換 PDF...")
    try:
        images = convert_pdf_to_images(pdf_path)
    except Exception as e:
        print(f"PDF轉換錯誤: {e}")
        return None
    
    if not images:
        print("PDF轉換失敗或PDF為空")
        return None
        
    # 顯示PDF總頁數並讓使用者設定分組
    total_pages = len(images)
    print(f"\n此PDF檔案共有 {total_pages} 頁")
    
    while True:
        try:
            pages_per_group = int(input("請輸入每組有幾頁: "))
            total_groups = (total_pages + pages_per_group - 1) // pages_per_group
            if pages_per_group > 0:
                print(f"總共有 {total_groups} 組")
                break
            else:
                print("每組頁數必須大於0")
        except ValueError:
            print("請輸入有效的數字！")
    
    while True:
        try:
            page_in_group = int(input(f"在每一組中，要分析第幾頁 (1-{pages_per_group}): "))
            if 1 <= page_in_group <= pages_per_group:
                break
            else:
                print(f"請輸入有效的頁碼 (1-{pages_per_group})")
        except ValueError:
            print("請輸入有效的數字！")
    
    all_page_results = []
    
    # 第一次圈選時儲存區域位置
    first_page = True
    saved_student_id_region = None
    saved_answer_regions = None
    
    # Initialize variables
    student_id_region = []
    answer_regions = []
    question_numbers = []
    correct_answers = []
    
    # 修改處理每組的迴圈
    for group in range(total_groups):
        page_to_process = group * pages_per_group + (page_in_group - 1)
        # 確保不會超出總頁數
        if page_to_process >= total_pages:
            break
        
        image = images[page_to_process]
        print(f"\n處理第 {group + 1} 組的第 {page_in_group} 頁 (PDF第 {page_to_process + 1} 頁)")
        
        # 確保圖像是 PIL Image 格式
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if first_page:
            print("\n請在圖片視窗中圈選座號區域")
            saved_student_id_region = select_regions(np.array(image))
            
            print("\n請在圖片視窗中圈選答案區域")
            saved_answer_regions = select_regions(np.array(image))
            
            # 輸入題號和正確答案
            question_numbers = []
            correct_answers = []
            for i in range(len(saved_answer_regions)):
                while True:
                    try:
                        num = input(f"請輸入題號 {i+1} 的題號: ")
                        ans = input(f"請輸入題號 {i+1} 的正確答案: ")
                        question_numbers.append(num)
                        correct_answers.append(ans)
                        break
                    except ValueError:
                        print("請輸入有效的題號和答案!")
            
            first_page = False
        
        # 使儲存的區域
        student_id_region = saved_student_id_region
        answer_regions = saved_answer_regions
        
        # 提取座號
        student_id = extract_student_id(np.array(image), student_id_region)
        print(f"辨識到的座號: {student_id}")
        
        # 提取答案
        results = extract_answers(image, answer_regions, question_numbers)
        
        # 創建標註用的圖像副本
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        try:
            font = ImageFont.load_default()
        except Exception as e:
            print(f"無法載入字型: {e}")
            font = None
        
        # 標註結果
        if student_id_region:  # 確保有座號區域
            for x1, y1, x2, y2 in student_id_region:
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                if font:
                    draw.text((x1, y1-20), f"座號: {student_id}", fill="blue", font=font)
        
        # 標註答案
        for i, ((x1, y1, x2, y2), ans) in enumerate(zip(answer_regions, results)):
            color = "green" if ans == correct_answers[i] else "red"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            if font:
                draw.text((x1, y1-20), f"Q{question_numbers[i]}: {ans}", fill=color, font=font)
        
        # 儲存標註後的圖片
        output_image_path = f"{os.path.splitext(pdf_path)[0]}_group{group+1}_annotated.jpg"
        annotated_image.save(output_image_path)
        print(f"已儲存標註結果至: {output_image_path}")
        
        # 準備結果
        answers_dict = {
            'student_id': student_id,
            'answers': [],
            'correct_count': 0
        }
        
        # 處理每題的答案和結果
        for i in range(min(len(results), len(question_numbers))):
            q_num = question_numbers[i]
            correct_ans = correct_answers[i]
            student_ans = results[i] if i < len(results) else ''
            is_correct = student_ans.strip() == correct_ans.strip()
            if is_correct:
                answers_dict['correct_count'] += 1
            answers_dict['answers'].append({
                'question': q_num,
                'student_answer': student_ans,
                'correct_answer': correct_ans,
                'is_correct': is_correct
            })
        
        all_page_results.append(answers_dict)
    
    # 取得 PDF 檔名（不含路徑和副檔名）
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # 建立輸出檔名
    output_filename = f"{pdf_filename}_第{page_in_group}頁.csv"
    
    # 儲存結果
    save_results_to_csv(all_page_results, output_filename)
    
    return all_page_results

def save_results_to_csv(all_results, output_filename='all_answers.csv'):
    # 獲取所有題號
    all_questions = sorted(list(set(
        ans['question'] for result in all_results 
        for ans in result['answers']
    )))
    
    # 準備 CSV 標頭
    headers = ['座號'] + all_questions + ['答對題數']
    
    # 準備每個學生的資料
    rows = []
    for result in all_results:
        row = {'座號': result['student_id']}
        
        # 填入學生的作答答案
        for ans in result['answers']:
            row[ans['question']] = ans['student_answer']
        
        # 加入答對題數
        row['答對題數'] = result['correct_count']
        rows.append(row)
    
    # 寫入 CSV
    with open(output_filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def select_pdf_files():
    root = tk.Tk()
    root.withdraw()  # 隱藏主窗口
    file_paths = filedialog.askopenfilenames(
        title="選擇 PDF 檔案",
        filetypes=[("PDF files", "*.pdf")]
    )
    return file_paths

if __name__ == "__main__":
    pdf_paths = select_pdf_files()
    
    if pdf_paths:
        all_results = []
        for pdf_path in pdf_paths:
            print(f"\n處理檔案: {pdf_path}")
            result = process_pdf(pdf_path)
            if result:  # 確保結果不是 None
                # 顯示結果
                print("\n處理完成！結果如下")
                for page_result in result:
                    if page_result:  # 確保頁面結果不是 None
                        print(f"\n座號: {page_result['student_id']}")
                        for ans in page_result['answers']:
                            print(f"{ans['question']}: 作答={ans['student_answer']}, "
                                  f"正確答案={ans['correct_answer']}")
                        print(f"答對題數: {page_result['correct_count']}")
            else:
                print(f"處理 {pdf_path} 時發生錯誤")
    else:
        print("未選擇檔案")

# 創建 debug 資料夾（如果不存在）
if not os.path.exists('debug'):
    os.makedirs('debug')
