import fitz  # PyMuPDF
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import json  # 在文件開頭添加

# 首先定義 ZoneSelector 類別
class ZoneSelector:
    def __init__(self, image_path):
        self.root = tk.Toplevel()
        self.root.title("選擇分析區域")
        
        # 載入圖片
        self.image = Image.open(image_path)
        # 調整圖片大小以適應螢幕
        screen_width = self.root.winfo_screenwidth() * 0.8
        screen_height = self.root.winfo_screenheight() * 0.8
        scale = min(screen_width/self.image.width, screen_height/self.image.height)
        new_width = int(self.image.width * scale)
        new_height = int(self.image.height * scale)
        self.image = self.image.resize((new_width, new_height))
        self.photo = ImageTk.PhotoImage(self.image)
        
        # 創建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 創建畫布
        self.canvas = tk.Canvas(self.main_frame, width=new_width, height=new_height)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        
        # 創建控制面板
        self.control_panel = ttk.Frame(self.main_frame)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 狀態標籤
        self.status_label = ttk.Label(self.control_panel, text="請點擊設定每個題目的分隔線")
        self.status_label.pack(pady=5)
        
        # 清單框顯示已選擇的區域
        self.zone_listbox = tk.Listbox(self.control_panel, width=30, height=10)
        self.zone_listbox.pack(pady=5)
        
        # 控制按鈕
        self.next_button = ttk.Button(self.control_panel, text="下一步", 
                                    command=self.next_step, state='disabled')
        self.next_button.pack(pady=5)
        
        self.clear_button = ttk.Button(self.control_panel, text="清除", 
                                     command=self.clear_selection)
        self.clear_button.pack(pady=5)
        
        self.finish_button = ttk.Button(self.control_panel, text="完成", 
                                      command=self.finish, state='disabled')
        self.finish_button.pack(pady=5)
        
        # 在控制面板中加入兩個滑動條框架
        self.range_frame_x = ttk.LabelFrame(self.control_panel, text="水平判定寬度")
        self.range_frame_x.pack(pady=5, padx=5, fill="x")
        
        self.range_frame_y = ttk.LabelFrame(self.control_panel, text="垂直判定高度")
        self.range_frame_y.pack(pady=5, padx=5, fill="x")
        
        # X方向的滑動條（用於分數判定）
        self.x_range = tk.IntVar(value=10)
        self.range_slider_x = ttk.Scale(
            self.range_frame_x,
            from_=5,
            to=50,
            variable=self.x_range,
            orient="horizontal",
            command=self.update_range
        )
        self.range_slider_x.pack(pady=5, padx=5, fill="x")
        self.range_label_x = ttk.Label(self.range_frame_x, text="水平寬度: 10")
        self.range_label_x.pack(pady=5)
        
        # Y方向的滑動條（用於題目高度判定）
        self.y_range = tk.IntVar(value=15)
        self.range_slider_y = ttk.Scale(
            self.range_frame_y,
            from_=5,
            to=50,
            variable=self.y_range,
            orient="horizontal",
            command=self.update_range
        )
        self.range_slider_y.pack(pady=5, padx=5, fill="x")
        self.range_label_y = ttk.Label(self.range_frame_y, text="垂直高度: 10")
        self.range_label_y.pack(pady=5)
        
        # 在控制面板添加匯出/匯入按鈕
        self.export_button = ttk.Button(self.control_panel, text="匯出設定", 
                                      command=self.export_settings)
        self.export_button.pack(pady=5)
        
        self.import_button = ttk.Button(self.control_panel, text="匯入設定", 
                                      command=self.import_settings)
        self.import_button.pack(pady=5)
        
        # 初始化變數
        self.selection_state = "SET_HEIGHTS"  # SET_HEIGHTS, SET_X_POSITIONS
        self.question_heights = []
        self.score_x_positions = []
        self.score_zones = []
        
        # 綁定事件
        self.canvas.bind("<Button-1>", self.handle_click)
        
        # 等待使用者操作
        self.root.wait_window()
    
    def update_range(self, *args):
        """更新判定範圍時重繪所有線條"""
        new_x_range = self.x_range.get()
        new_y_range = self.y_range.get()
        self.range_label_x.config(text=f"水平寬度: {new_x_range}")
        self.range_label_y.config(text=f"垂直高度: {new_y_range}")
        
        # 清除所有現有的線條
        self.canvas.delete("range_lines")
        
        # 重繪分數位置的線條
        if self.selection_state == "SET_X_POSITIONS" and self.score_x_positions:
            for x_pos in self.score_x_positions:
                # 中心線
                self.canvas.create_line(
                    x_pos, 0, x_pos, self.image.height,
                    fill="green", width=2, tags="range_lines"
                )
                # 左右範圍線
                self.canvas.create_line(
                    x_pos - new_x_range, 0, x_pos - new_x_range, self.image.height,
                    fill="green", dash=(4, 4), tags="range_lines"
                )
                self.canvas.create_line(
                    x_pos + new_x_range, 0, x_pos + new_x_range, self.image.height,
                    fill="green", dash=(4, 4), tags="range_lines"
                )
        
        # 重繪題目分隔線和中間的判定線
        sorted_heights = sorted(self.question_heights)
        
        # 先畫所有分隔線
        for y_pos in sorted_heights:
            self.canvas.create_line(
                0, y_pos, self.image.width, y_pos,
                fill="blue", width=2, tags="range_lines"
            )
        
        # 再畫所有中間判定線
        for i in range(len(sorted_heights) - 1):
            y1 = sorted_heights[i]
            y2 = sorted_heights[i + 1]
            mid_y = (y1 + y2) // 2
            
            # 畫中間的判定線（虛線）
            self.canvas.create_line(
                0, mid_y, self.image.width, mid_y,
                fill="red", dash=(4, 4), tags="range_lines"
            )
            # 畫上下範圍線
            self.canvas.create_line(
                0, mid_y - new_y_range, self.image.width, mid_y - new_y_range,
                fill="red", dash=(4, 4), tags="range_lines"
            )
            self.canvas.create_line(
                0, mid_y + new_y_range, self.image.width, mid_y + new_y_range,
                fill="red", dash=(4, 4), tags="range_lines"
            )
    
    def update_score_zones(self):
        """更新分數區域的範圍"""
        self.score_zones.clear()
        sorted_heights = sorted(self.question_heights)
        sorted_x_positions = sorted(self.score_x_positions)
        
        # 對每個分數位置
        for i, x_pos in enumerate(sorted_x_positions):
            score = i + 1
            # 對每個題目區域
            for j in range(len(sorted_heights) - 1):
                y1 = sorted_heights[j]
                y2 = sorted_heights[j + 1]
                mid_y = (y1 + y2) // 2  # 計算中點
                
                # 修改判定區域的位置，將其上移
                y_offset = (y2 - y1) // 2  # 使用1/4的區域高度作為偏移
                
                self.score_zones.append({
                    'x1': x_pos - self.x_range.get(),
                    'y1': mid_y - y_offset - self.y_range.get(),  # 上移判定區域
                    'y2': mid_y - y_offset + self.y_range.get(),  # 上移判定區域
                    'x2': x_pos + self.x_range.get(),
                    'score': score,
                    'question': j + 1
                })
    
    def handle_click(self, event):
        if self.selection_state == "SET_HEIGHTS":
            y_pos = event.y
            self.question_heights.append(y_pos)
            
            # 畫出中心線和範圍線
            current_y_range = self.y_range.get()
            self.canvas.create_line(
                0, y_pos, self.image.width, y_pos,
                fill="blue", width=2, tags="range_lines"
            )
            self.canvas.create_line(
                0, y_pos - current_y_range, self.image.width, y_pos - current_y_range,
                fill="blue", dash=(4, 4), tags="range_lines"
            )
            self.canvas.create_line(
                0, y_pos + current_y_range, self.image.width, y_pos + current_y_range,
                fill="blue", dash=(4, 4), tags="range_lines"
            )
            
            self.zone_listbox.insert(tk.END, f"題目分隔線: y={y_pos}")
            self.next_button['state'] = 'normal'
        
        elif self.selection_state == "SET_X_POSITIONS":
            x_pos = event.x
            self.score_x_positions.append(x_pos)
            current_range = self.x_range.get()
            
            # 畫出中心線和範圍線
            self.canvas.create_line(
                x_pos, 0, x_pos, self.image.height,
                fill="green", width=2, tags="range_lines"
            )
            self.canvas.create_line(
                x_pos - current_range, 0, x_pos - current_range, self.image.height,
                fill="green", dash=(4, 4), tags="range_lines"
            )
            self.canvas.create_line(
                x_pos + current_range, 0, x_pos + current_range, self.image.height,
                fill="green", dash=(4, 4), tags="range_lines"
            )
            
            self.zone_listbox.insert(tk.END, f"分數X軸位置: x={x_pos}")
            
            # 更新分數區域
            self.update_score_zones()
            
            if len(self.score_x_positions) > 0:
                self.finish_button['state'] = 'normal'
    
    def next_step(self):
        if self.selection_state == "SET_HEIGHTS":
            self.selection_state = "SET_X_POSITIONS"
            self.status_label["text"] = "請點擊每個分數欄位的中心位置"
            self.next_button['state'] = 'disabled'
    
    def clear_selection(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.zone_listbox.delete(0, tk.END)
        self.question_heights = []
        self.score_zones = []
        self.selection_state = "SET_HEIGHTS"
        self.status_label["text"] = "請點擊設定每個題目的分隔線"
        self.next_button['state'] = 'disabled'
        self.finish_button['state'] = 'disabled'
    
    def finish(self):
        self.root.destroy()
    
    def get_zones(self):
        question_intervals = []
        # 確保 question_heights 已排序
        sorted_heights = sorted(self.question_heights)
        for i in range(len(sorted_heights) - 1):
            question_intervals.append({
                'y1': int(sorted_heights[i]),      # 確保是整數
                'y2': int(sorted_heights[i + 1])   # 確保是整數
            })
        return question_intervals, self.score_zones
    
    def export_settings(self):
        """匯出設定到JSON檔案"""
        settings = {
            'question_heights': self.question_heights,
            'score_x_positions': self.score_x_positions,
            'x_range': self.x_range.get(),
            'y_range': self.y_range.get()
        }
        
        file_path = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="儲存設定檔"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
            tk.messagebox.showinfo("成功", "設定已成功匯出！")
    
    def import_settings(self):
        """從JSON檔案匯入設定"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="選擇設定檔"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # 清除現有設定
                self.clear_selection()
                
                # 載入設定
                self.question_heights = settings['question_heights']
                self.score_x_positions = settings['score_x_positions']
                self.x_range.set(settings['x_range'])
                self.y_range.set(settings['y_range'])
                
                # 更新UI
                for y_pos in self.question_heights:
                    self.zone_listbox.insert(tk.END, f"題目分隔線: y={y_pos}")
                
                for x_pos in self.score_x_positions:
                    self.zone_listbox.insert(tk.END, f"分數X軸位置: x={x_pos}")
                
                # 更新狀態
                if self.question_heights:
                    self.selection_state = "SET_X_POSITIONS"
                    self.status_label["text"] = "請點擊每個分數欄位的中心位置"
                    self.next_button['state'] = 'normal'
                
                if self.score_x_positions:
                    self.finish_button['state'] = 'normal'
                
                # 重新繪製所有線條
                self.update_range()
                self.update_score_zones()
                
                tk.messagebox.showinfo("成功", "設定已成功匯入！")
            
            except Exception as e:
                tk.messagebox.showerror("錯誤", f"匯入設定時發生錯誤：{str(e)}")

# 然後是 process_page 函數
def process_page(page, csv_writer, question_intervals, score_zones):
    # 獲取圖片
    pix = page.get_pixmap()
    output_image_path = f"page_{page.number + 1}.png"
    pix.save(output_image_path)

    # 使用OpenCV讀取影像
    image = cv2.imread(output_image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 2)

    # 使用形態學與去除格線
    kernel_length = max(binary_image.shape[1] // 100, binary_image.shape[0] // 100)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    vertical_lines_img = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    horizontal_lines_img = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    binary_image_no_lines = cv2.subtract(binary_image, horizontal_lines_img)
    binary_image_no_lines = cv2.subtract(binary_image_no_lines, vertical_lines_img)

    # 尋找二值圖中的勾勾位置
    contours, _ = cv2.findContours(binary_image_no_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 尋找特定比例的物件
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.3 < aspect_ratio < 2.5 and 3 < cv2.contourArea(contour) < 500:
            filtered_contours.append(contour)

    # 初始化每個題目的分數和勾選計數器
    question_scores = ["" for _ in range(len(question_intervals))]
    question_mark_count = [0 for _ in range(len(question_intervals))]  
    question_selected_scores = [[] for _ in range(len(question_intervals))]  # 新增：記錄每題選擇的分數

    # 檢查每個勾選位置是否在判定區域內
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_center_x = x + w // 2
        contour_center_y = y + h // 2

        # 檢查每個判定區域
        for zone in score_zones:
            if (zone['x1'] <= contour_center_x <= zone['x2'] and 
                zone['y1'] <= contour_center_y <= zone['y2']):
                question_number = zone['question'] - 1
                if question_number < len(question_scores):
                    question_mark_count[question_number] += 1
                    question_selected_scores[question_number].append(str(zone['score']))
                    if question_mark_count[question_number] > 1:
                        # 檢查是否所有選擇的分數都相同
                        if len(set(question_selected_scores[question_number])) == 1:
                            question_scores[question_number] = question_selected_scores[question_number][0]
                        else:
                            question_scores[question_number] = "???"
                    else:
                        question_scores[question_number] = str(zone['score'])

    # 顯示結果的部分
    image_with_scores_and_boxes = cv2.cvtColor(binary_image_no_lines, cv2.COLOR_GRAY2BGR)
    overlay = image_with_scores_and_boxes.copy()
    
    # 在圖片上畫出所有判定區域
    for zone in score_zones:
        # 畫出半透明的判定區域（淺藍色）
        cv2.rectangle(overlay, 
                     (int(zone['x1']), int(zone['y1'])), 
                     (int(zone['x2']), int(zone['y2'])), 
                     (255, 200, 0), -1)
        
        # 畫出判定框框（深藍色）
        cv2.rectangle(image_with_scores_and_boxes, 
                     (int(zone['x1']), int(zone['y1'])), 
                     (int(zone['x2']), int(zone['y2'])), 
                     (255, 0, 0), 1)
        
        # 在區域上方顯示分數選項
        cv2.putText(image_with_scores_and_boxes, 
                   str(zone['score']), 
                   (int(zone['x1']), int(zone['y1'])-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 將半透明覆蓋層與原圖合併
    alpha = 0.3
    image_with_scores_and_boxes = cv2.addWeighted(overlay, alpha, 
                                                 image_with_scores_and_boxes, 
                                                 1 - alpha, 0)

    # 在檢測到勾選的地方添加標記
    overlay = image_with_scores_and_boxes.copy()  # 創建一個覆蓋層
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_center_x = x + w // 2
        contour_center_y = y + h // 2
        
        # 提示中心點位置
        cv2.circle(image_with_scores_and_boxes, 
                  (contour_center_x, contour_center_y), 
                  3, (0, 255, 0), -1)
        
        # 在覆蓋層上繪製綠色框
        cv2.rectangle(overlay, 
                     (x, y), (x+w, y+h), 
                     (0, 255, 0), 2)
    
    # 將覆蓋層與原圖合併
    alpha = 0.3  # 透明度：0是完全透明，1是完全不透明
    image_with_scores_and_boxes = cv2.addWeighted(overlay, alpha, 
                                                 image_with_scores_and_boxes, 
                                                 1 - alpha, 0)

    # 在圖片上顯示結果時，使用不同顏色標示多重勾選和未作答
    for i, score in enumerate(question_scores):
        y_pos = (question_intervals[i]['y1'] + question_intervals[i]['y2']) // 2
        score_text = f"Q{i+1}: {score if score else 'no answer'}"
        
        # 根據勾選情況選擇顏色
        if question_mark_count[i] > 1:
            color = (100, 100, 255)  # 多重勾選用紅色
        elif not score:  # 未作答
            color = (255, 192, 203)  # 粉色 (BGR格式)
        else:
            color = (255, 255, 100)  # 單一勾選用綠色
            
        cv2.putText(image_with_scores_and_boxes,
                   score_text,
                   (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,
                   color,
                   2)

    # 保存結果圖片
    cv2.imwrite(f"page_{page.number + 1}_with_scores.png", image_with_scores_and_boxes)

    # 寫入 CSV
    page_data = [page.number + 1] + question_scores
    csv_writer.writerow(page_data)


# 修改最後的主程式部分
def main():
    print("請選擇PDF檔案")
    pdf_path = filedialog.askopenfilename(
        title="選擇PDF檔案",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )

    if not pdf_path:
        print("未選擇檔案，程式結束")
        exit()

    pages_per_group = int(input("請輸入每張數: "))
    specific_page = int(input(f"請輸入要分析每組中的第幾張(1-{pages_per_group}): "))

    if specific_page < 1 or specific_page > pages_per_group:
        print(f"錯誤：每組只有{pages_per_group}張，無法選擇第{specific_page}張")
        exit()

    # 開啟 PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    total_groups = total_pages // pages_per_group

    print(f"PDF共有 {total_pages} 頁，可分為 {total_groups} 組")
    
    # 讓使用者選擇要用哪一組來設定或是匯入設定
    #print("\n請選擇操作：")
    #print("1. 使用PDF頁面來設定範本")
    #print("2. 匯入已存在的設定檔")
    # choice = input("請輸入選項 (1/2): ")
    choice = "1"
    question_intervals = None
    score_zones = None

    if choice == "1":
        #group_to_setup = int(input(f"請選擇要用第幾組來設定範本 (1-{total_groups}): ")) - 1
        group_to_setup = 0
        template_page_number = group_to_setup * pages_per_group + (specific_page - 1)
        template_page = doc.load_page(template_page_number)
        
        pix = template_page.get_pixmap()
        pix.save("temp_template_page.png")
        
        root = tk.Tk()
        root.withdraw()
        selector = ZoneSelector("temp_template_page.png")
        question_intervals, score_zones = selector.get_zones()
        os.remove("temp_template_page.png")
    
    elif choice == "2":
        settings_path = filedialog.askopenfilename(
            title="選擇設定檔",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if settings_path:
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                
            # 建立臨時的ZoneSelector來處理設定
            root = tk.Tk()
            root.withdraw()
            selector = ZoneSelector("temp_template_page.png")  # 這裡不會實際使用到圖片
            selector.question_heights = settings['question_heights']
            selector.score_x_positions = settings['score_x_positions']
            selector.x_range.set(settings['x_range'])
            selector.y_range.set(settings['y_range'])
            selector.update_score_zones()
            question_intervals, score_zones = selector.get_zones()

    # 寫入 CSV
    with open('scores_data.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = ['第幾頁'] + [f'Q{i}' for i in range(1, len(question_intervals) + 1)]
        csv_writer.writerow(headers)

        # 處理每組中的特定頁面
        for group in range(total_groups):
            page_number = group * pages_per_group + (specific_page - 1)
            if page_number < total_pages:
                page = doc.load_page(page_number)
                process_page(page, csv_writer, question_intervals, score_zones)
                print(f"已處理 {group + 1} 組的第 {specific_page} 張（第 {page_number + 1} 頁）")

    print("分析完成")
    doc.close()

if __name__ == "__main__":
    main()