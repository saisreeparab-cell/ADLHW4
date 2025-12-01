from pathlib import Path
import json
import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    karts = extract_kart_objects(info_path, view_index, img_width = img_width, img_height = img_height)
    track = extract_track_info(info_path)

    captions = []

    if not karts:
        captions.append(f"A scene on the {track} track with no visible karts.")
        return captions
    # find ego
    ego = next((k for k in karts if k.get("is_center_kart")), karts[0])
    captions.append(f"{ego['kart_name']} is the ego car on the {track} track. " f"There are {len(karts)} karts visible.")

    other = [k for k in karts if k["instance_id"] != ego["instance_id"]]
    ex, ey = ego["center"]
    for o in other[:2]:
        # determine position word
        #ex, ey = ego["center"]
        ox, oy = o["center"]
        lr = "left" if ox < ex else "right"
        fb = "front" if oy < ey else "back"
        captions.append(f"{o['kart_name']} is {lr} and {fb} the ego car.")

    return captions

    # raise NotImplementedError("Not implemented")


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

def build_train(split: str = "train"):
   
    base_data_dir = Path(__file__).parent.parent / "data"
    info_dir = base_data_dir / split
    out_file = info_dir / "tux_captions.json"

    all_caps = []

    for info_path in sorted(info_dir.glob("*_info.json")):
        with open(info_path) as f:
            info = json.load(f)

        num_views = len(info["detections"])
        base_name = info_path.stem.replace("_info", "")

        for view_index in range(num_views):
            captions = generate_caption(str(info_path), view_index)

            # Same convention as QA: RELATIVE to data/
            image_file = f"{split}/{base_name}_{view_index:02d}_im.jpg"

            for cap in captions:
                all_caps.append(
                    {
                        "image_file": image_file,
                        "caption": cap,
                    }
                )

    print(f"Total caption pairs for split '{split}': {len(all_caps)}")
    with open(out_file, "w") as f:
        json.dump(all_caps, f, indent=2)
    print(f"Saved to {out_file}")


def main():
    fire.Fire(
        {
            "check": check_caption,
            "build_train": build_train,
        }
    )


if __name__ == "__main__":
    main()

