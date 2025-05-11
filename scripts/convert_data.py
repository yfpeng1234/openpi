import os
import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

import robosuite.utils.transform_utils as T

def load_dataset(dataset_path):
    f = h5py.File(dataset_path, "r")
    return f

def main(data_dir: str, output_dir: str, task_id: int, *, push_to_hub: bool = False):
    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    REPO_NAME="DexMG/task{}".format(task_id)
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="DexMG",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (84, 84, 3),
                "names": ["height", "width", "channel"],
            },
            "left_wrist_image": {
                "dtype": "image",
                "shape": (84, 84, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_image": {
                "dtype": "image",
                "shape": (84, 84, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (36,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (24,),
                "names": ["actions"],
            },
        },
        image_writer_threads=20,
        image_writer_processes=10,
        root=os.path.join(output_dir, 'task{}'.format(task_id)),
    )

    # load the original dataset
    original_dataset = load_dataset(data_dir)
    demo_keys=list(original_dataset["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demo_keys])
    demo_id = [demo_keys[i] for i in inds]

    # language instructions
    if task_id==1:
        task_name='move the box lid onto the box'
    elif task_id==3:
        task_name='put the two objects in the tray and then lift the tray'
    else:
        raise ValueError("task_id should be 1 or 3")

    # start conversion
    for demo_idx in demo_id:
        episode= original_dataset["data"][demo_idx]
        for time_idx in range(episode['actions'].shape[0]):
            dataset.add_frame(
                {
                    "image": episode["obs"]["agentview_image"][time_idx],
                    "left_wrist_image": episode["obs"]["robot1_eye_in_hand_image"][time_idx],
                    "right_wrist_image": episode["obs"]["robot0_eye_in_hand_image"][time_idx],
                    "state": np.concatenate([episode['obs']['robot0_eef_pos'][time_idx], 
                                             T.quat2axisangle(episode['obs']['robot0_eef_quat'][time_idx]),
                                             episode['obs']['robot0_gripper_qpos'][time_idx],
                                             episode['obs']['robot1_eef_pos'][time_idx],
                                             T.quat2axisangle(episode['obs']['robot1_eef_quat'][time_idx]),
                                             episode['obs']['robot1_gripper_qpos'][time_idx]
                                             ]).astype(np.float32),
                    "actions": episode["actions"][time_idx].astype(np.float32),
                }
            )
        dataset.save_episode(task=task_name)

    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)