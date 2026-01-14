import sys
import os
import json
import random
import requests
import urllib.parse
import io
import re
import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Updated to use the new Google Gen AI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("CRITICAL ERROR: The 'google.genai' package is missing.")
    print("Please run: pip install google-genai")
    sys.exit(1)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog, 
    QProgressBar, QTextEdit, QGroupBox, QFormLayout, QSpinBox,
    QTabWidget, QSplitter, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# ================= CONFIGURATION DEFAULTS =================
CONFIG_FILE = "frtools_config.json"

DEFAULT_CONFIG = {
    "pollin_api_key": "",
    "gemini_api_key": "",
    "output_dir": os.path.join(os.getcwd(), "frtools_assets_v5"),
    "model": "flux",
    "gemini_model": "gemini-2.5-flash",
    "width": 1024,
    "height": 1024,
    "batch_size": 5,
    "start_index": 0,
    "last_batch_file": "ioun_stones.json"
}

POLLIN_MODELS = [
    "kontext", "turbo", "nanobanana", "nanobanana-pro", 
    "seedream", "seedream-pro", "gptimage", "gptimage-large", 
    "flux", "zimage"
]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview"
]

# ================= HTML STYLES FOR 3.5E CARDS =================
# This CSS mimics the official 3.5e Dungeon Master's Guide visual style
DND_35_STYLE = """
<style>
    body {
        font-family: 'Book Antiqua', 'Palatino Linotype', serif;
        background-color: #f4e4bc;
        color: #000;
        margin: 20px;
        background-image: repeating-linear-gradient(0deg, transparent, transparent 19px, #e6d3a3 20px);
    }
    .stat-block {
        background: #fff8dc;
        border: 2px solid #5c4033;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.5);
        padding: 15px;
        max-width: 700px;
        margin: auto;
        position: relative;
    }
    h1 {
        font-family: 'DnD', sans-serif;
        color: #800000;
        border-bottom: 2px solid #800000;
        font-size: 24px;
        margin-top: 0;
        clear: none;
    }
    .type-tag {
        font-style: italic;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .item-image {
        float: right;
        width: 250px;
        height: 250px;
        border: 3px solid #5c4033;
        margin-left: 15px;
        margin-bottom: 10px;
        object-fit: contain;
        background-color: #fff;
        border-radius: 8px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 13px;
    }
    th {
        background-color: #d2b48c;
        text-align: left;
        padding: 4px;
        border: 1px solid #5c4033;
        font-weight: bold;
    }
    td {
        padding: 4px;
        border: 1px solid #5c4033;
        background-color: #faebd7;
    }
    .description {
        margin-top: 15px;
        font-size: 14px;
        line-height: 1.4;
        text-align: justify;
    }
    .physical-description {
        margin-top: 10px;
        font-size: 13px;
        font-style: italic;
        color: #333;
        background-color: rgba(255,255,255,0.3);
        padding: 5px;
        border: 1px dashed #8b4513;
    }
    .cost-block {
        margin-top: 15px;
        font-weight: bold;
        color: #800000;
        text-align: right;
        clear: both;
    }
    h2 {
        font-size: 16px;
        color: #5c4033;
        border-bottom: 1px solid #5c4033;
        margin-top: 10px;
    }
</style>
"""

# ================= WORKER THREAD =================
class GenerationWorker(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, items_data, config, mode="batch"):
        super().__init__()
        self.raw_items_data = items_data
        self.config = config
        self.mode = mode # "batch" or "single"
        self._is_running = True
        self.client = None

    def run(self):
        # Configure Gemini Client
        if not self.config['gemini_api_key']:
            self.log_signal.emit("Error: Gemini API Key is missing!")
            self.finished_signal.emit()
            return

        try:
            self.client = genai.Client(api_key=self.config['gemini_api_key'])
            model_name = self.config.get('gemini_model', 'gemini-2.5-flash')
            self.log_signal.emit(f"Connected to {model_name} (google.genai).")
        except Exception as e:
            self.log_signal.emit(f"Gemini Init Error: {e}")
            self.finished_signal.emit()
            return

        # Prepare Batch
        items_to_process = []
        if self.mode == "single":
            # In single mode, raw_items_data is a single dict object
            items_to_process.append(("Manual", self.raw_items_data))
        else:
            # Flatten dictionary to list of (category, item) tuples
            # Expected format: {"CategoryName": [ {item1}, {item2} ]}
            all_items_flat = []
            if isinstance(self.raw_items_data, dict):
                for cat, items in self.raw_items_data.items():
                    if isinstance(items, list):
                        for item in items:
                            all_items_flat.append((cat, item))
            
            # Apply Batch Logic
            start_idx = self.config.get('start_index', 0)
            batch_size = self.config.get('batch_size', len(all_items_flat))
            
            # Bounds checking
            if start_idx >= len(all_items_flat):
                self.log_signal.emit("Start Index exceeds total items available.")
                self.finished_signal.emit()
                return

            items_to_process = all_items_flat[start_idx : start_idx + batch_size]
        
        total_to_process = len(items_to_process)
        if total_to_process == 0:
            self.log_signal.emit("Batch empty or invalid structure.")
            self.finished_signal.emit()
            return

        self.log_signal.emit(f"Starting job: {total_to_process} item(s)...")

        for i, (category, item) in enumerate(items_to_process):
            if not self._is_running: break
            
            item_name = item.get('name', 'Unknown Item')
            self.log_signal.emit(f"Forging: {item_name}...")
            
            try:
                # 1. Generate Content via Gemini
                content_data = self.generate_content_with_gemini(item)
                
                # 2. Forge Image
                self.log_signal.emit(f"  > Creating Visual Token...")
                
                # Determine filenames ahead of time
                safe_name = "".join([c for c in item_name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(" ", "_")
                image_filename = f"{safe_name}.png"
                
                # Extract image prompt from Gemini response
                img_prompt = content_data.get('image_prompt', item_name)
                
                # Actually generate the image
                if self.process_image(img_prompt, category, safe_name):
                    # 3. Inject Image into HTML and Save
                    html_content = content_data.get('html_card', '<p>Error generating stats.</p>')
                    
                    # HTML injection for the image
                    img_tag = f'<img src="{image_filename}" class="item-image" alt="{item_name}">'
                    
                    # Add image tag at start of content
                    if '<div class="physical-description">' in html_content:
                        # Insert before description if possible
                        final_html_content = html_content.replace('<div class="physical-description">', f'{img_tag}\n<div class="physical-description">', 1)
                    else:
                        final_html_content = img_tag + html_content
                    
                    self.save_html_card(category, item_name, safe_name, final_html_content)
                    self.log_signal.emit(f"  > Success: {item_name}")
                else:
                    self.log_signal.emit(f"  > Image Generation Failed: {item_name}")
                    self.save_html_card(category, item_name, safe_name, content_data['html_card'])

            except Exception as e:
                self.log_signal.emit(f"  > Error processing {item_name}: {str(e)}")

            # Update progress relative to the batch
            self.progress_signal.emit(int(((i + 1) / total_to_process) * 100))
            self.msleep(500)
            
        self.finished_signal.emit()

    def generate_content_with_gemini(self, item):
        """Uses selected Gemini model to generate both the prompt and the HTML card."""
        name = item.get('name', 'Unknown')
        itype = item.get('type', 'Wondrous Item')
        val = item.get('val', 'Unknown gp')
        user_desc = item.get('description', '')
        
        prompt = (
            f"You are a D&D 3.5e Forgotten Realms expert. I need comprehensive data for the magic item '{name}' (Type: {itype}, Value: {val}).\n"
        )
        
        if user_desc:
            prompt += f"USER MANUAL DESCRIPTION/CONTEXT: '{user_desc}'. USE THIS TO GUIDE THE FLAVOR AND STATS.\n"
        
        prompt += (
            "Return a pure JSON object with exactly two keys:\n"
            "1. 'image_prompt': A highly detailed stable diffusion prompt for a 'Top-down view, orthographic, isolated "
            "RPG item token'. Include specific materials (mithral, gold, leather), magical effects (glowing runes, mist), "
            "and color palette suitable for the item's name. Focus on physical description. Do NOT include 'background' instructions.\n"
            "2. 'html_card': A complete HTML snippet (NO <html>, <head>, or <body> tags). "
            "It must contain:\n"
            "   - A detailed 'Physical Description' div (class='physical-description') describing exactly what the item looks like.\n"
            "   - A 'Stats' table including: Caster Level, Prerequisites, Market Price, Cost to Create, Weight, Aura.\n"
            "   - A 'Description' section (class='description') detailing its magical powers, history, and activation methods.\n"
            "   - A 'Construction' section if applicable.\n"
            "Do not add the <h1> title."
        )

        try:
            model_name = self.config.get('gemini_model', 'gemini-2.5-flash')
            
            # Use JSON enforcement in API config to prevent syntax errors
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            text = response.text if hasattr(response, 'text') else str(response)
            
            # Robust JSON extraction and cleanup
            # Even with response_mime_type, sometimes we want to be safe
            text = re.sub(r"```json|```", "", text).strip()
            
            # Attempt parsing
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Fallback: Sometimes specific chars break it, try to isolate the brace block
                if "{" in text:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    text = text[start:end]
                    return json.loads(text)
                else:
                    raise

        except Exception as e:
            # Pass original error up to worker log
            raise Exception(f"Gemini Generation Failed: {e}")

    def save_html_card(self, category, name, safe_name, html_content):
        out_path = os.path.join(self.config['output_dir'], category)
        os.makedirs(out_path, exist_ok=True)
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{name}</title>
            {DND_35_STYLE}
        </head>
        <body>
            <div class="stat-block">
                <h1>{name}</h1>
                <div class="type-tag">Magic Item ({category})</div>
                <hr>
                {html_content}
                <div class="cost-block">Item Ref: {safe_name}</div>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(out_path, f"{safe_name}.html"), 'w', encoding='utf-8') as f:
            f.write(full_html)

    def advanced_transparency(self, image_bytes):
        """Advanced background removal using thresholding and edge smoothing."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            arr = np.array(img)
            
            # Mask for near-white or flat background colors (Pollinations usually gives white/black BG)
            r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
            # Aggressive white removal
            mask = (r > 240) & (g > 240) & (b > 240)
            arr[mask, 3] = 0 
            
            processed_img = Image.fromarray(arr)
            
            # Edge cleanup
            alpha = processed_img.getchannel('A')
            alpha = alpha.filter(ImageFilter.MedianFilter(3)) 
            processed_img.putalpha(alpha)
            
            return processed_img
        except Exception as e:
            print(f"Transparency Error: {e}")
            return Image.open(io.BytesIO(image_bytes))

    def process_image(self, visual_description, category, safe_name):
        # Hardcoded Prompt Construction to force transparency friendly generation
        final_prompt = (
            f"{visual_description}. "
            f"white background, isolated on white, negative space, cut-out, 8k resolution, "
            f"professional game asset, sharp focus, top-down orthographic projection."
        )

        encoded_prompt = urllib.parse.quote(final_prompt)
        
        # User specified endpoint for paid tier/rate limit fix
        # Using https://gen.pollinations.ai/image/{prompt} instead of image.pollinations.ai
        url = f"https://gen.pollinations.ai/image/{encoded_prompt}"
        
        params = {
            "width": self.config['width'], 
            "height": self.config['height'], 
            "nologo": "true", 
            "model": self.config['model'], 
            "seed": random.randint(1, 10**9),
            "enhance": "false"
        }
        
        headers = {}
        if self.config.get('pollin_api_key'):
            # If a key is present, we include it in the Authorization header 
            # as specifically requested for the paid tier.
            headers["Authorization"] = f"Bearer {self.config['pollin_api_key']}"
            params["private"] = "true"
        
        try:
            res = requests.get(url, params=params, headers=headers, timeout=120)
            if res.status_code == 200:
                processed_img = self.advanced_transparency(res.content)
                
                out_path = os.path.join(self.config['output_dir'], category)
                os.makedirs(out_path, exist_ok=True)
                
                img_name = f"{safe_name}.png"
                processed_img.save(os.path.join(out_path, img_name), "PNG")
                return True
            else:
                self.log_signal.emit(f"  > API Error: {res.status_code} - {res.text}")
        except Exception as e: 
            self.log_signal.emit(f"  > API Error: {str(e)}")
        return False

# ================= MAIN WINDOW =================
class FRAssetGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FRTools Magic Item Forge (Batch + Manual)")
        self.resize(1100, 850)
        self.config = self.load_cfg()
        self.init_ui()
    
    def load_cfg(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f: return {**DEFAULT_CONFIG, **json.load(f)}
            except: pass
        return DEFAULT_CONFIG

    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central); layout = QVBoxLayout(central)
        
        # --- API Settings ---
        auth = QGroupBox("API Keys"); a_form = QFormLayout()
        self.p_in = QLineEdit(self.config['pollin_api_key']); self.p_in.setEchoMode(QLineEdit.EchoMode.Password)
        self.g_in = QLineEdit(self.config['gemini_api_key']); self.g_in.setEchoMode(QLineEdit.EchoMode.Password)
        a_form.addRow("Pollinations API (Optional):", self.p_in)
        a_form.addRow("Gemini API (Required):", self.g_in)
        auth.setLayout(a_form); layout.addWidget(auth)

        # --- Model Settings ---
        model_grp = QGroupBox("AI Model Settings"); m_layout = QHBoxLayout()
        
        self.gemini_sel = QComboBox(); self.gemini_sel.addItems(GEMINI_MODELS)
        self.gemini_sel.setCurrentText(self.config.get('gemini_model', 'gemini-2.5-flash'))
        
        self.model_sel = QComboBox(); self.model_sel.addItems(POLLIN_MODELS)
        self.model_sel.setCurrentText(self.config['model'])
        
        self.size_sel = QComboBox(); self.size_sel.addItems(["512", "768", "1024", "1280", "2048"])
        self.size_sel.setCurrentText(str(self.config['width']))
        
        m_layout.addWidget(QLabel("Gemini Model (Text):")); m_layout.addWidget(self.gemini_sel)
        m_layout.addWidget(QLabel("Image Model:")); m_layout.addWidget(self.model_sel)
        m_layout.addWidget(QLabel("Resolution:")); m_layout.addWidget(self.size_sel)
        model_grp.setLayout(m_layout); layout.addWidget(model_grp)

        # --- Tabs for Modes ---
        tabs = QTabWidget()
        
        # TAB 1: BATCH FORGE
        batch_tab = QWidget()
        b_main_layout = QVBoxLayout(batch_tab)
        
        batch_grp = QGroupBox("Batch Processing Source"); b_layout = QVBoxLayout()
        
        # File Selection Row
        file_row = QHBoxLayout()
        self.batch_file_display = QLineEdit(self.config.get("last_batch_file", "ioun_stones.json"))
        btn_batch_file = QPushButton("Select JSON File")
        btn_batch_file.clicked.connect(self.select_batch_file)
        file_row.addWidget(QLabel("Source File:")); file_row.addWidget(self.batch_file_display); file_row.addWidget(btn_batch_file)
        b_layout.addLayout(file_row)

        # Batch Settings Row
        settings_row = QHBoxLayout()
        self.start_spin = QSpinBox(); self.start_spin.setRange(0, 10000); 
        self.start_spin.setValue(self.config.get('start_index', 0))
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 500); 
        self.batch_spin.setValue(self.config.get('batch_size', 5))
        settings_row.addWidget(QLabel("Start Index:")); settings_row.addWidget(self.start_spin)
        settings_row.addWidget(QLabel("Batch Size:")); settings_row.addWidget(self.batch_spin)
        b_layout.addLayout(settings_row)

        batch_grp.setLayout(b_layout)
        
        self.batch_btn = QPushButton("FORGE BATCH ASSETS"); self.batch_btn.clicked.connect(self.start_batch)
        self.batch_btn.setFixedHeight(50)
        self.batch_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 14px;")
        
        b_main_layout.addWidget(batch_grp)
        b_main_layout.addWidget(self.batch_btn)
        b_main_layout.addStretch()
        
        # TAB 2: MANUAL FORGE
        manual_tab = QWidget()
        m_main_layout = QVBoxLayout(manual_tab)
        
        manual_grp = QGroupBox("Single Item Details"); m_form = QFormLayout()
        self.m_name = QLineEdit()
        self.m_name.setPlaceholderText("e.g. Ring of Winter")
        self.m_type = QLineEdit()
        self.m_type.setPlaceholderText("e.g. Ring (Artifact)")
        self.m_val = QLineEdit()
        self.m_val.setPlaceholderText("e.g. 50,000 gp")
        self.m_desc = QTextEdit()
        self.m_desc.setPlaceholderText("Describe the item here. Mention appearance, powers, and history. The AI will use this to generate the stats and image.")
        self.m_desc.setMaximumHeight(100)
        
        m_form.addRow("Item Name:", self.m_name)
        m_form.addRow("Item Type:", self.m_type)
        m_form.addRow("Market Value:", self.m_val)
        m_form.addRow("Description/Prompt:", self.m_desc)
        manual_grp.setLayout(m_form)
        
        self.manual_btn = QPushButton("FORGE SINGLE ITEM"); self.manual_btn.clicked.connect(self.start_manual)
        self.manual_btn.setFixedHeight(50)
        self.manual_btn.setStyleSheet("background-color: #8B0000; color: white; font-weight: bold; font-size: 14px;")
        
        m_main_layout.addWidget(manual_grp)
        m_main_layout.addWidget(self.manual_btn)
        
        tabs.addTab(batch_tab, "Batch Mode")
        tabs.addTab(manual_tab, "Manual Mode")
        layout.addWidget(tabs)

        # --- Output Settings ---
        dir_grp = QGroupBox("Storage Path"); d_layout = QHBoxLayout()
        self.dir_display = QLineEdit(self.config['output_dir'])
        btn_dir = QPushButton("Browse"); btn_dir.clicked.connect(self.select_dir)
        d_layout.addWidget(self.dir_display); d_layout.addWidget(btn_dir)
        dir_grp.setLayout(d_layout); layout.addWidget(dir_grp)

        # --- Logs & Progress ---
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet("background-color: #1a1a1a; color: #00e676; font-family: Consolas;")
        self.prog = QProgressBar()
        
        layout.addWidget(self.log); layout.addWidget(self.prog)

    def select_batch_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select JSON Batch File", "", "JSON Files (*.json)")
        if file:
            self.batch_file_display.setText(file)

    def select_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if d: self.dir_display.setText(d)

    def update_config(self):
        self.config.update({
            "pollin_api_key": self.p_in.text(),
            "gemini_api_key": self.g_in.text(),
            "output_dir": self.dir_display.text(),
            "model": self.model_sel.currentText(),
            "gemini_model": self.gemini_sel.currentText(),
            "width": int(self.size_sel.currentText()),
            "height": int(self.size_sel.currentText()),
            "start_index": self.start_spin.value(),
            "batch_size": self.batch_spin.value(),
            "last_batch_file": self.batch_file_display.text()
        })
        with open(CONFIG_FILE, 'w') as f: json.dump(self.config, f)

    def start_batch(self):
        batch_filename = self.batch_file_display.text()
        
        if not os.path.exists(batch_filename):
            self.log.append(f"Error: File '{batch_filename}' not found!")
            return
            
        try:
            with open(batch_filename, 'r', encoding='utf-8') as f:
                loot_data = json.load(f)
        except Exception as e:
            self.log.append(f"Error loading JSON file: {str(e)}")
            return

        if not loot_data: 
            self.log.append("Error: Batch file is empty or invalid JSON!")
            return
        
        self.update_config()
        self.toggle_ui(False)
        
        self.worker = GenerationWorker(loot_data, self.config, mode="batch")
        self.connect_worker()
        self.worker.start()

    def start_manual(self):
        if not self.m_name.text():
            self.log.append("Error: Item Name is required for manual forge.")
            return

        self.update_config()
        self.toggle_ui(False)
        
        item_data = {
            "name": self.m_name.text(),
            "type": self.m_type.text() or "Wondrous Item",
            "val": self.m_val.text() or "Unique",
            "description": self.m_desc.toPlainText()
        }
        
        self.worker = GenerationWorker(item_data, self.config, mode="single")
        self.connect_worker()
        self.worker.start()

    def connect_worker(self):
        self.worker.log_signal.connect(self.log.append)
        self.worker.progress_signal.connect(self.prog.setValue)
        self.worker.finished_signal.connect(lambda: self.toggle_ui(True))

    def toggle_ui(self, enabled):
        self.batch_btn.setEnabled(enabled)
        self.manual_btn.setEnabled(enabled)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FRAssetGenerator()
    win.show()
    sys.exit(app.exec())