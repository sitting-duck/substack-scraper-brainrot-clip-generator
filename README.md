# substack-scraper-brainrot-clip-generator
Scrapes substack articles and generates 30 second videos based on content to basically snag doomscrollers and direct them to the substack. Uses stock footage, generated scripts, and generated voiceover. Still rough but basically fully automated.


```bash
python -m venv .venv
source ./.venv/bin/activate

#once
pip install -r ./requirements.txt

python 1_fetch_substack.py
python 2_build_index.py
python 3_generate_scripts.py
python 4_prepare_for_videos.py
python 5_make_video.py

```
