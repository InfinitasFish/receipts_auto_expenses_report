ML pipeline that's built on fine-tuned LayoutXLM model and easy-ocr engine for extracting useful entities from receipts and fill expenses exel-report with them.


Инструкция по использованию:

Скачать poppler и добавить его в PATH
https://github.com/oschwartz10612/poppler-windows/releases/

Загрузить С++ build tools (нужны для детектрона)
https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/

Создать venv, установить все библиотеки с помощью requirements.txt

Скачать чекпоинт модели, переименовать папку в layoutXLM_checkpoint и закинуть в папку проекта
https://drive.google.com/drive/folders/1catnQYa9TfGX62cMEIufd2Q7R9R2WXnV?usp=sharing

Запустить flask_app.py
