from pathlib import Path

f = Path("ml/video_inspector.py")
t = f.read_text(encoding="utf-8")
dup = '''            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",'''
single = '''            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",'''
t = t.replace(dup, single)
f.write_text(t, encoding="utf-8")
print("Cleaned duplicate -vf")