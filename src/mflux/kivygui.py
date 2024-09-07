import os
import sys
import time
import asyncio
import threading
from queue import Queue
from datetime import datetime
import sqlite3
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.scrollview import ScrollView
from kivy.uix.progressbar import ProgressBar
from kivy.properties import StringProperty, BooleanProperty, ObjectProperty, ListProperty, NumericProperty
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.popup import Popup
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.checkbox import CheckBox 
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mflux.config.model_config import ModelConfig
from mflux.config.config import Config
from mflux.flux.flux import Flux1


class LabeledInput(BoxLayout):
    label_text = StringProperty('')
    input_filter = ObjectProperty(None)
    multiline = BooleanProperty(False)
    hint_text = StringProperty('')
    input_text = StringProperty('')

    def __init__(self, **kwargs):
        super(LabeledInput, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.add_widget(Label(text=self.label_text, size_hint_y=None, height='30dp'))
        self.text_input = TextInput(multiline=self.multiline, input_filter=self.input_filter,
                                    hint_text=self.hint_text, text=self.input_text)
        self.add_widget(self.text_input)

class ModelToggle(BoxLayout):
    def __init__(self, **kwargs):
        super(ModelToggle, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.add_widget(Label(text='Model', size_hint_y=None, height='30dp'))
        toggle_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height='40dp')
        self.schnell_button = ToggleButton(text='schnell', group='model', state='down')
        self.dev_button = ToggleButton(text='dev', group='model')
        toggle_layout.add_widget(self.schnell_button)
        toggle_layout.add_widget(self.dev_button)
        self.add_widget(toggle_layout)

    @property
    def selected_model(self):
        return 'schnell' if self.schnell_button.state == 'down' else 'dev'

class QuantizeToggle(BoxLayout):
    def __init__(self, **kwargs):
        super(QuantizeToggle, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.add_widget(Label(text='Quantize', size_hint_y=None, height='30dp'))
        toggle_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height='40dp')
        self.none_button = ToggleButton(text='None', group='quantize', state='down')
        self.four_button = ToggleButton(text='4', group='quantize')
        self.eight_button = ToggleButton(text='8', group='quantize')
        toggle_layout.add_widget(self.none_button)
        toggle_layout.add_widget(self.four_button)
        toggle_layout.add_widget(self.eight_button)
        self.add_widget(toggle_layout)

    @property
    def selected_quantize(self):
        if self.four_button.state == 'down':
            return 4
        elif self.eight_button.state == 'down':
            return 8
        else:
            return None

    def set_quantize(self, value):
        if value == 4:
            self.four_button.state = 'down'
            self.none_button.state = 'normal'
            self.eight_button.state = 'normal'
        elif value == 8:
            self.eight_button.state = 'down'
            self.none_button.state = 'normal'
            self.four_button.state = 'normal'
        else:
            self.none_button.state = 'down'
            self.eight_button.state = 'normal'
            self.four_button.state = 'normal'            

class ScrollableLabel(ScrollView):
    text = StringProperty('')

class ConfirmPopup(Popup):
    def __init__(self, confirm_callback, **kwargs):
        super(ConfirmPopup, self).__init__(**kwargs)
        self.confirm_callback = confirm_callback
        self.title = 'Confirm delete'
        self.size_hint = (0.6, 0.3)
        
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text='Do you want to delete this element ?'))
        
        buttons = BoxLayout(size_hint_y=None, height=44)
        cancel_button = Button(text='Cancel')
        cancel_button.bind(on_press=self.dismiss)
        confirm_button = Button(text='Confirm')
        confirm_button.bind(on_press=self.on_confirm)
        
        buttons.add_widget(cancel_button)
        buttons.add_widget(confirm_button)
        content.add_widget(buttons)
        
        self.content = content

    def on_confirm(self, *args):
        self.confirm_callback()
        self.dismiss()

class RequestHistoryItem(ButtonBehavior, BoxLayout, RecycleDataViewBehavior):
    prompt = StringProperty()
    model = StringProperty()
    timestamp = StringProperty()
    index = NumericProperty()

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        self.prompt = data['prompt']
        self.model = data['model']
        self.timestamp = data['timestamp']

    def on_press(self):
        app = App.get_running_app()
        app.root.load_request(self.index)

    def on_delete_press(self):
        app = App.get_running_app()
        app.root.show_delete_confirmation(self.index)

Builder.load_string('''
<RequestHistoryItem>:
    orientation: 'horizontal'
    padding: 5
    spacing: 2
    BoxLayout:
        orientation: 'vertical'
        size_hint_x: 0.8
        Label:
            text: f"Prompt: {root.prompt[:50]}..."
        Label:
            text: f"Model: {root.model}"
        Label:
            text: f"Date: {root.timestamp}"
    BoxLayout:
        orientation: 'vertical'
        size_hint_x: 0.2
        Button:
            text: 'Delete'
            on_press: root.on_delete_press()
            size_hint_y: None
            height: '40dp'

<RequestHistoryView>:
    viewclass: 'RequestHistoryItem'
    RecycleBoxLayout:
        default_size: None, dp(100)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'
''')

class RequestHistoryView(RecycleView):
    def __init__(self, **kwargs):
        super(RequestHistoryView, self).__init__(**kwargs)
        self.data = []

class ImageGeneratorGUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.padding = 10
        self.spacing = 10

        # Initialize database
        self.init_database()

        self.db_queue = Queue()
        Clock.schedule_interval(self.process_db_queue, 0.1)

        # Left panel for controls
        left_scroll = ScrollView(size_hint_x=0.3)
        self.left_panel = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None)
        self.left_panel.bind(minimum_height=self.left_panel.setter('height'))
        left_scroll.add_widget(self.left_panel)
        self.setup_left_panel()

        # Center panel for image display
        self.center_panel = BoxLayout(orientation='vertical', size_hint_x=0.4)
        self.image_widget = Image(size_hint=(1, 1))
        self.center_panel.add_widget(self.image_widget)
        
        # Right panel for request history
        self.right_panel = BoxLayout(orientation='vertical', size_hint_x=0.3)
        self.setup_right_panel()

        # Add all panels to the main layout
        self.add_widget(left_scroll)
        self.add_widget(self.center_panel)
        self.add_widget(self.right_panel)

        # Load request history
        self.load_request_history()


    def setup_left_panel(self):
        self.left_panel.add_widget(Label(text='Flux image generator', size_hint_y=None, height='40dp'))

        self.prompt_input = TextInput(hint_text='Enter your prompt here', multiline=True, size_hint_y=None, height='150dp')  # Increased height for 5 lines
        self.left_panel.add_widget(self.prompt_input)

        self.model_toggle = ModelToggle(size_hint_y=None, height='70dp')
        self.model_toggle.schnell_button.bind(on_press=self.on_model_toggle)
        self.model_toggle.dev_button.bind(on_press=self.on_model_toggle)
        self.left_panel.add_widget(self.model_toggle)

        ss_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height='70dp')
        self.seed_input = LabeledInput(label_text='Seed', hint_text='Random if empty', input_filter='int')
        self.steps_input = LabeledInput(label_text='Steps', input_text='4', input_filter='int')
        ss_layout.add_widget(self.seed_input)
        ss_layout.add_widget(self.steps_input)
        self.left_panel.add_widget(ss_layout)

        hw_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height='70dp')
        self.height_input = LabeledInput(label_text='Height', input_text='1024', input_filter='int')
        self.width_input = LabeledInput(label_text='Width', input_text='1024', input_filter='int')
        hw_layout.add_widget(self.height_input)
        hw_layout.add_widget(self.width_input)
        self.left_panel.add_widget(hw_layout)

        self.guidance_input = LabeledInput(label_text='Guidance', input_text='3.5', input_filter='float', size_hint_y=None, height='70dp')
        self.left_panel.add_widget(self.guidance_input)
        
        self.quantize_toggle = QuantizeToggle(size_hint_y=None, height='70dp')
        self.left_panel.add_widget(self.quantize_toggle)

        self.path_input = LabeledInput(label_text='Model Path', hint_text='Local path for loading model', size_hint_y=None, height='70dp')
        self.left_panel.add_widget(self.path_input)

        self.lora_paths_input = LabeledInput(label_text='LORA Paths', hint_text='Space-separated paths', size_hint_y=None, height='70dp')
        self.left_panel.add_widget(self.lora_paths_input)

        self.lora_scales_input = LabeledInput(label_text='LORA Scales', hint_text='Space-separated scales', size_hint_y=None, height='70dp')
        self.left_panel.add_widget(self.lora_scales_input)

        metadata_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height='40dp')
        self.metadata_checkbox = CheckBox(size_hint_x=None, width='30dp')
        metadata_label = Label(text='Export image metadata as JSON')
        metadata_layout.add_widget(self.metadata_checkbox)
        metadata_layout.add_widget(metadata_label)
        self.left_panel.add_widget(metadata_layout)

        self.generate_button = Button(text='Generate picture', size_hint_y=None, height='40dp')
        self.generate_button.bind(on_press=self.start_generation)
        self.left_panel.add_widget(self.generate_button)

        self.progress_bar = ProgressBar(max=100, size_hint_y=None, height='20dp')
        self.left_panel.add_widget(self.progress_bar)

        self.progress_label = Label(text='', size_hint_y=None, height='30dp')
        self.left_panel.add_widget(self.progress_label)

        self.output_label = Label(text='', size_hint_y=None, height='100dp')
        self.left_panel.add_widget(self.output_label)

    def setup_right_panel(self):
        self.right_panel.add_widget(Label(text='Query history', size_hint_y=None, height='40dp'))
        self.history_view = RequestHistoryView()
        self.right_panel.add_widget(self.history_view)

    def init_database(self):
        self.conn = sqlite3.connect('request_history.db')
        self.cursor = self.conn.cursor()
        
        # Vérifier si la table existe déjà
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='requests'")
        table_exists = self.cursor.fetchone()
        
        if not table_exists:
            # Si la table n'existe pas, créez-la avec tous les champs
            self.cursor.execute('''
            CREATE TABLE requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                model TEXT,
                seed INTEGER,
                steps INTEGER,
                height INTEGER,
                width INTEGER,
                guidance REAL,
                quantize INTEGER,
                path TEXT,
                lora_paths TEXT,
                lora_scales TEXT,
                export_metadata BOOLEAN,
                timestamp DATETIME
            )
            ''')
        else:
            # Si la table existe, vérifiez si les nouveaux champs sont présents, sinon ajoutez-les
            existing_columns = [column[1] for column in self.cursor.execute("PRAGMA table_info(requests)").fetchall()]
            
            new_columns = {
                'quantize': 'INTEGER',
                'path': 'TEXT',
                'lora_paths': 'TEXT',
                'lora_scales': 'TEXT',
                'export_metadata': 'BOOLEAN'
            }
            
            for column, data_type in new_columns.items():
                if column not in existing_columns:
                    self.cursor.execute(f"ALTER TABLE requests ADD COLUMN {column} {data_type}")
        
        self.conn.commit()

    def show_delete_confirmation(self, index):
        content = ConfirmPopup(confirm_callback=lambda: self.delete_request(index))
        content.open()

    def save_request(self, prompt, model, seed, steps, height, width, guidance, quantize, path, lora_paths, lora_scales, export_metadata):
        def db_save_request():
            timestamp = datetime.now()
            self.cursor.execute('''
            INSERT INTO requests (prompt, model, seed, steps, height, width, guidance, quantize, path, lora_paths, lora_scales, export_metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (prompt, model, seed, steps, height, width, guidance, quantize, path, lora_paths, lora_scales, export_metadata, timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")))
            self.conn.commit()
            return timestamp

        result_queue = Queue()
        self.db_queue.put((db_save_request, (), lambda result: result_queue.put(result)))
        timestamp = result_queue.get()
        
        Clock.schedule_once(lambda dt: self.load_request_history())
        
        return timestamp
    
    def load_request_history(self, *args):
        def db_load_request_history():
            self.cursor.execute("SELECT prompt,model,timestamp, id FROM requests ORDER BY timestamp DESC LIMIT 50")
            return self.cursor.fetchall()

        def update_history(requests):
            self.history_view.data = [
                {
                    'prompt': row[0],
                    'model': row[1],
                    'timestamp': str(row[2]),
                    'index': str(row[3])
                } for idx, row in enumerate(requests)
            ]

        self.db_queue.put((db_load_request_history, (), update_history))
        
    def load_request(self, index):
        self.cursor.execute("SELECT id, prompt, model, seed, steps, height, width, guidance, quantize, path, lora_paths, lora_scales, export_metadata, timestamp FROM `requests` ORDER BY timestamp DESC LIMIT 50")
        column_names = [description[0] for description in self.cursor.description]
        requests = self.cursor.fetchall()
        if 0 <= index < len(requests):
            request = dict(zip(column_names, requests[index]))
            
            self.prompt_input.text = request.get('prompt', '')
            self.model_toggle.schnell_button.state = 'down' if request.get('model') == 'schnell' else 'normal'
            self.model_toggle.dev_button.state = 'down' if request.get('model') == 'dev' else 'normal'
            self.seed_input.text_input.text = str(request.get('seed', ''))
            self.steps_input.text_input.text = str(request.get('steps', ''))
            self.height_input.text_input.text = str(request.get('height', ''))
            self.width_input.text_input.text = str(request.get('width', ''))
            self.guidance_input.text_input.text = str(request.get('guidance', ''))
            
            self.quantize_toggle.set_quantize(request.get('quantize'))
            path = request.get('path', '')
            self.path_input.text_input.text = path if path else ''
            lora_paths = request.get('lora_paths', '')
            self.lora_paths_input.text_input.text = lora_paths if lora_paths else ''
            lora_scales = request.get('lora_scales', '')
            self.lora_paths_input.text_input.text = lora_scales if lora_scales else ''

            self.metadata_checkbox.active = bool(request.get('export_metadata', False))

            # Chargement de l'image générée
            timestamp = request.get('timestamp')
            if timestamp:
                try:
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                    image_filename = f'./pictures/generated_image_{timestamp.strftime("%Y%m%d_%H%M%S")}.png'
                    if os.path.exists(image_filename):
                        self.image_widget.source = image_filename
                        self.image_widget.reload()
                    else:
                        print(f"File not found: {image_filename}")
                        self.image_widget.source = ''  # Supprimer l'image précédente si non trouvée
                except ValueError:
                    print(f"Invalid timestamp format: {timestamp}")
                    self.image_widget.source = ''
            else:
                print("No timestamp available for this request")
                self.image_widget.source = ''


    def on_model_toggle(self, instance):
        if self.model_toggle.selected_model == 'dev':
            if self.guidance_input not in self.left_panel.children:
                self.left_panel.add_widget(self.guidance_input, index=len(self.left_panel.children)-4)
        else:
            if self.guidance_input in self.left_panel.children:
                self.left_panel.remove_widget(self.guidance_input)

    def start_generation(self, instance):
        self.generate_button.disabled = True
        self.output_label.text = 'Starting generation...'
        self.progress_bar.value = 0
        self.progress_label.text = ''
        threading.Thread(target=self.generate_image_thread, daemon=True).start()

    def run_async_generate_image(self):
        asyncio.run(self.generate_image())

    def generate_image_thread(self):

        prompt = self.prompt_input.text
        model = self.model_toggle.selected_model
        seed = int(self.seed_input.text_input.text) if self.seed_input.text_input.text else int(time.time())
        height = int(self.height_input.text_input.text)
        width = int(self.width_input.text_input.text)
        steps = int(self.steps_input.text_input.text)
        guidance = float(self.guidance_input.text_input.text) if model == 'dev' and self.guidance_input.text_input.text else 3.5
        quantize = self.quantize_toggle.selected_quantize
        path = self.path_input.text_input.text if self.path_input.text_input.text else None
        lora_paths = self.lora_paths_input.text_input.text.split() if self.lora_paths_input.text_input.text else None
        lora_scales = [float(scale) for scale in self.lora_scales_input.text_input.text.split()] if self.lora_scales_input.text_input.text else None
        export_metadata = self.metadata_checkbox.active


        # Sauvegarde de la requête dans la base de données
        timestamp = self.save_request(prompt, model, seed, steps, height, width, guidance, quantize, path, lora_paths, lora_scales, export_metadata)

        Clock.schedule_once(lambda dt: self.update_output('Init model...'))
        try:
            # Initialisation du modèle Flux1
            flux = Flux1(
                model_config=ModelConfig.from_alias(model),
                quantize=quantize,
                local_path=path,
                lora_paths=lora_paths,
                lora_scales=lora_scales
            )
        except Exception as e:
            Clock.schedule_once(lambda dt: self.update_output(f"Error initializing model: {str(e)}"))
            return

        Clock.schedule_once(lambda dt: self.update_output('Generate picture...'))

        start_time = time.time()
        last_update_time = start_time
        total_time = 0

                
        def update_progress(current_step, total_steps):
            nonlocal last_update_time, total_time
            current_time = time.time()
            iteration_time = current_time - last_update_time
            total_time += iteration_time
            last_update_time = current_time

            progress = (current_step / total_steps) * 100
            avg_time_per_step = total_time / current_step if current_step > 0 else 0
            remaining_steps = total_steps - current_step
            estimated_time_remaining = avg_time_per_step * remaining_steps

            status = (
                f'Step: {current_step}/{total_steps}\n'
                f'Time per iteration: {iteration_time:.2f}s\n'
                f'Estimated remaining time: {estimated_time_remaining:.2f}s'
            )

            Clock.schedule_once(lambda dt: self.update_progress(status, progress))
            
        config = Config(
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance
        )

        image = flux.generate_image(
            seed=seed,
            prompt=prompt,
            config=config,
            progress_callback=update_progress
        )

        Clock.schedule_once(lambda dt: self.update_output('Saving picture...'))
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        output_filename = f'./pictures/generated_image_{timestamp_str}.png'
        
        # Save the image
        image.save(path=output_filename, export_json_metadata=export_metadata)

        Clock.schedule_once(lambda dt: self.update_image(output_filename, seed))
        
        # Ajouter cette ligne pour recharger l'historique après la génération
        Clock.schedule_once(lambda dt: self.load_request_history())

    async def async_run(self, func, *args, **kwargs):
        future = asyncio.Future()
        def run_in_thread():
            try:
                result = func(*args, **kwargs)
                asyncio.run_coroutine_threadsafe(future.set_result(result), asyncio.get_event_loop())
            except Exception as e:
                asyncio.run_coroutine_threadsafe(future.set_exception(e), asyncio.get_event_loop())
        threading.Thread(target=run_in_thread).start()
        return await future

    async def async_update_output(self, message):
        Clock.schedule_once(lambda dt: setattr(self.output_label, 'text', self.output_label.text + f'\n{message}'))

    async def async_update_progress(self, message, value=None):
        Clock.schedule_once(lambda dt: self.update_progress(message, value))

    async def async_update_image(self, output_filename, seed, metadata_status):
        Clock.schedule_once(lambda dt: self.update_image(output_filename, seed, metadata_status))
    def update_output(self, message):
        self.output_label.text += f'\n{message}'

    def update_progress(self, message, value=None):
        self.progress_label.text = message
        if value is not None:
            self.progress_bar.value = value

    def update_image(self, output_filename, seed):
        self.image_widget.source = output_filename
        self.image_widget.reload()
        self.output_label.text += f'Picture generated successfully: {output_filename}'
        self.output_label.text += f'\nSeed : {seed}'
        self.generate_button.disabled = False
        self.progress_label.text = 'Generation finished'
        self.progress_bar.value = 100

    def process_db_queue(self, dt):
        while not self.db_queue.empty():
            task, args, callback = self.db_queue.get()
            result = task(*args)
            if callback:
                if isinstance(callback, tuple):
                    func, *callback_args = callback
                    func(result, *callback_args)
                else:
                    callback(result)

    #def load_request_history(self, *args):
    #    def db_load_request_history():
    #        self.cursor.execute("SELECT * FROM requests ORDER BY timestamp DESC LIMIT 50")
    #        return self.cursor.fetchall()

    #    def update_history(requests):
    #        self.history_view.data = [
    #            {
    #                'prompt': row[1],
    #                'model': row[2],
    #                'timestamp': str(row[8]),
    #                'index': idx
    #            } for idx, row in enumerate(requests)
    #        ]

    #    self.db_queue.put((db_load_request_history, (), update_history))

    def delete_request(self, index):
        def db_delete_request(index):
            self.cursor.execute("SELECT id FROM requests ORDER BY timestamp DESC LIMIT 50")
            request_ids = self.cursor.fetchall()
            if 0 <= index < len(request_ids):
                request_id = request_ids[index][0]
                self.cursor.execute("DELETE FROM requests WHERE id = ?", (request_id,))
                self.conn.commit()

        self.db_queue.put((db_delete_request, (index,), (lambda x: self.load_request_history(),)))


class ImageGeneratorApp(App):
    def build(self):
        Window.size = (1500, 800)
        return ImageGeneratorGUI()

if __name__ == '__main__':
    ImageGeneratorApp().run()