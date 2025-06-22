# MemoriesOrganizer

This tool organizes photos and videos by extracting metadata and using a large language model (LLM) to generate descriptive names.

## Features

1. Indexes a directory containing photos and videos.
2. Extracts EXIF information including modification time and GPS location.
3. For videos, extracts frames every 30 seconds.
4. Sends gathered information to the LLM and receives a short descriptive name using structured output.
5. Renames the files with the generated name.

## Requirements

- Python 3.8+
- `opencv-python`
- `pillow`
- `langchain`
- `openai`
- `numpy`
- `piexif`

## Usage

```bash
pip install -r requirements.txt
python -m memories_organizer.main <directory>
```

An OpenAI API key must be available in the environment as `OPENAI_API_KEY` for the LLM calls.

## Running Tests

```bash
pytest -q
```

The tests generate small sample media files at runtime to validate frame extraction and metadata parsing.
