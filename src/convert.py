import glob
import os
import shutil
from collections import defaultdict

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    dataset_path = "/home/alex/DATASETS/TODO/archive"
    batch_size = 30

    def create_ann(image_path):
        labels = []
        tags = []

        subfolder = image_path.split("/")[-3]
        seq = sly.Tag(sequence_meta, value=subfolder)
        tags.append(seq)

        # image_np = sly.imaging.image.read(image_path)[:, :, 0]
        if "Synth" in image_path:
            img_height = 544  # image_np.shape[0]
            img_wight = 960  # image_np.shape[1]
            synth_meta = synth_to_meta[image_path.split("/")[-5]]
            synth = sly.Tag(synth_meta)
            tags.append(synth)
            subfolder = image_path.split("/")[-5] + "^" + image_path.split("/")[-3]

        else:
            img_height = 1080
            img_wight = 1920

        bboxes_data = folder_to_gt_data[subfolder][get_file_name_with_ext(image_path)]
        if len(bboxes_data) > 1:
            for curr_bboxes_data in bboxes_data:
                l_tags = []
                target_id_tag = sly.Tag(target_id_meta, value=int(curr_bboxes_data[0]))
                l_tags.append(target_id_tag)

                obj_class = index_to_class[int(curr_bboxes_data[6])]

                left = float(curr_bboxes_data[1])
                right = left + float(curr_bboxes_data[3])
                top = float(curr_bboxes_data[2])
                bottom = top + float(curr_bboxes_data[4])

                if top > bottom or left > right:
                    continue

                rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                label = sly.Label(rectangle, obj_class, tags=l_tags)
                labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    fish = sly.ObjClass("fish", sly.Rectangle)
    crab = sly.ObjClass("crab", sly.Rectangle)
    shrimp = sly.ObjClass("shrimp", sly.Rectangle)
    starfish = sly.ObjClass("starfish", sly.Rectangle)
    small = sly.ObjClass("small fish", sly.Rectangle)
    jellyfish = sly.ObjClass("jellyfish", sly.Rectangle)

    index_to_class = {1: fish, 2: crab, 3: shrimp, 4: starfish, 5: small, 6: jellyfish}

    sequence_meta = sly.TagMeta("sequence", sly.TagValueType.ANY_STRING)
    target_id_meta = sly.TagMeta("target id", sly.TagValueType.ANY_NUMBER)

    synth = sly.TagMeta("plain background, no turbidity, no distractors", sly.TagValueType.NONE)
    synth_b = sly.TagMeta("video background, no turbidity, no distractors", sly.TagValueType.NONE)

    synth_t = sly.TagMeta("plain background, turbidity, no distractors", sly.TagValueType.NONE)
    synth_d = sly.TagMeta("plain background, no turbidity, with distractors", sly.TagValueType.NONE)
    synth_dt = sly.TagMeta("plain background, with turbidityand distractors", sly.TagValueType.NONE)
    synth_bt = sly.TagMeta(
        "video background with turbidity, but without distractors", sly.TagValueType.NONE
    )
    synth_bd = sly.TagMeta(
        "video background with distractor, but no turbidity", sly.TagValueType.NONE
    )
    synth_btd = sly.TagMeta(
        "video background with turbidity and distractors", sly.TagValueType.NONE
    )

    synth_to_meta = {
        "brackishMOTSynth": synth,
        "brackishMOTSynth_B": synth_b,
        "brackishMOTSynth_BD": synth_bd,
        "brackishMOTSynth_BF": synth_bt,
        "brackishMOTSynth_BFD": synth_btd,
        "brackishMOTSynth_D": synth_d,
        "brackishMOTSynth_F": synth_t,
        "brackishMOTSynth_FD": synth_dt,
    }

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=list(index_to_class.values()),
        tag_metas=[
            sequence_meta,
            target_id_meta,
            synth,
            synth_b,
            synth_bd,
            synth_bt,
            synth_btd,
            synth_d,
            synth_t,
            synth_dt,
        ],
    )
    api.project.update_meta(project.id, meta.to_json())

    train_images_pathes_n = glob.glob(dataset_path + "/BrackishMOT/train/*/img1/*.jpg")
    test_images_pathes_n = glob.glob(dataset_path + "/BrackishMOT/test/*/img1/*.jpg")
    train_images_pathes_s = glob.glob(dataset_path + "/brackishMOTSynth/*/train/*/img1/*.jpg")
    train_images_pathes = train_images_pathes_s + train_images_pathes_n

    folder_to_gt_data = {}
    gt_pathes_n = glob.glob(dataset_path + "/BrackishMOT/*/*/gt/gt.txt")
    gt_pathes_s = glob.glob(dataset_path + "/brackishMOTSynth/*/*/*/gt/gt.txt")
    gt_pathes = gt_pathes_n + gt_pathes_s
    for curr_path in gt_pathes:
        folder = curr_path.split("/")[-3]
        if "Synth" in curr_path:
            folder = curr_path.split("/")[-5] + "^" + curr_path.split("/")[-3]
        temp_dict = defaultdict(list)
        with open(curr_path) as f:
            content = f.read().split("\n")
            for row in content:
                if len(row) > 1:
                    curr_data = row.split(",")
                    im_name = curr_data[0].zfill(6) + ".jpg"
                    temp_dict[im_name].append(curr_data[1:])

        folder_to_gt_data[folder] = temp_dict

    ds_name_to_data = {"train": train_images_pathes, "test": test_images_pathes_n}

    for ds_name, images_pathes in ds_name_to_data.items():

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

        for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
            img_names_batch = []
            for im_path in img_pathes_batch:
                if "Synth" in im_path:
                    prefix = im_path.split("/")[-5]
                    if prefix == "brackishMOTSynth":
                        im_name = im_path.split("/")[-3] + "_" + get_file_name_with_ext(im_path)
                    else:
                        im_name = (
                            prefix.split("_")[1]
                            + "_"
                            + im_path.split("/")[-3]
                            + "_"
                            + get_file_name_with_ext(im_path)
                        )
                else:
                    im_name = im_path.split("/")[-3] + "_" + get_file_name_with_ext(im_path)

                img_names_batch.append(im_name)

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(img_names_batch))

    return project
