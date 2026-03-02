import os
import io

from sam3.detection_module import run_sam3_detection

import cv2
from PIL import Image
import numpy as np
import json
from final_prompt import prompt_template
import time
from google import genai
from google.genai import types
import io
import re


class SAM_VLM_Planner:

    def __init__(self):
        pass
    
    def process_videos_with_tags(self, directories, performance_tags, n_frames=5, max_videos=None):
        """
        Process video directories and build a results dict of coalesced frame images.

        max_videos: int, list of ints, or None.
        - None  : use all videos in every directory
        - int   : use at most that many videos from every directory
        - list  : one limit per directory, e.g. [3, 5]
        """

        if len(directories) != len(performance_tags):
            raise ValueError(
                f"Number of directories ({len(directories)}) must match "
                f"number of tags ({len(performance_tags)})"
            )

        # Normalise max_videos to a list aligned with directories
        if max_videos is None:
            limits = [None] * len(directories)
        elif isinstance(max_videos, int):
            limits = [max_videos] * len(directories)
        else:
            if len(max_videos) != len(directories):
                raise ValueError(
                    f"max_videos list length ({len(max_videos)}) must match "
                    f"number of directories ({len(directories)})"
                )
            limits = list(max_videos)

        results = {}

        def subsample_video_frames(video_path, n_frames):
            """Extract n equally spaced frames from a video."""
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                cap.release()
                raise ValueError(f"Video has no frames: {video_path}")

            if n_frames >= total_frames:
                indices = list(range(total_frames))
            else:
                indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (224, 224))
                    frames.append(frame)

            cap.release()
            return frames

        def coalesce_frames(frames):
            """Combine frames horizontally into a single panoramic image."""
            if not frames:
                raise ValueError("No frames to coalesce")
            return np.hstack(frames)

        print(f"Processing {len(directories)} directories...")

        for directory, tag, limit in zip(directories, performance_tags, limits):
            if tag.lower() not in ['success', 'failure']:
                raise ValueError(f"Invalid tag: {tag}. Must be 'success' or 'failure'")

            if not os.path.exists(directory):
                print(f"Warning: Directory not found: {directory}, skipping...")
                continue

            video_files = sorted([f for f in os.listdir(directory) if f.endswith('.mp4') and f.startswith('front_img')])

            if not video_files:
                print(f"Warning: No videos found in {directory}")
                continue

            print(f"\nProcessing directory: {directory} (tag: {tag})")
            print(f"  Found {len(video_files)} video(s)")

            selected = sorted(video_files)
            if limit is not None:
                selected = selected[:limit]
                print(f"  Using {len(selected)}/{len(video_files)} video(s) (max_videos={limit})")

            for video_file in selected:
                video_path = os.path.join(directory, video_file)
                wrist_path = os.path.join(directory, video_file.replace('front_img', 'wrist_img'))

                front_frames = subsample_video_frames(video_path, n_frames)
                wrist_frames = subsample_video_frames(wrist_path, n_frames)
                # Interleave: front_0, wrist_0, front_1, wrist_1, ...
                frames = [f for pair in zip(front_frames, wrist_frames) for f in pair]

                if not frames:
                    print(f"  ✗ No frames extracted from {video_file}")
                    continue

                # Unique key: parent_dir + dir_name + base_name to avoid collisions
                base_name = os.path.splitext(video_file)[0]
                dir_name = os.path.basename(directory)
                parent_name = os.path.basename(os.path.dirname(directory))
                key = f"{parent_name}_{dir_name}_{base_name}"

                coalesced_image = coalesce_frames(frames)
                coalesced_image_rgb = cv2.cvtColor(coalesced_image, cv2.COLOR_BGR2RGB)

                results[key] = {
                    'image': coalesced_image_rgb,
                    'tag': tag.lower(),
                }
                print(f"  ✓ {video_file}: {coalesced_image_rgb.shape[1]}x{coalesced_image_rgb.shape[0]} (tag: {tag})")

        print(f"\n{'='*60}")
        print(f"Total videos processed: {len(results)}")
        success_count = sum(1 for v in results.values() if v['tag'] == 'success')
        failure_count = sum(1 for v in results.values() if v['tag'] == 'failure')
        print(f"  Success: {success_count}")
        print(f"  Failure: {failure_count}")
        print(f"{'='*60}\n")

        return results
    

    def send_to_vlm(self, results, task_instruction, new_initial_state,
                    prompt_template=None, model_id="gemini-3.1-pro-preview",
                    save_responses=True, output_file='vlm_responses.json', output_dir=None):
        """Send coalesced rollout images plus a new observation to the VLM for analysis."""

        if not results:
            print("No results to send to VLM")
            return {}

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        vlm_client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(temperature=0.3)

        total = len(results)

        print(f"\n{'='*60}")
        print(f"Sending all {total} images in a single VLM call ({model_id})...")
        print(f"{'='*60}\n")

        # Build contents list: label + image for each rollout, then current observation
        # Remap keys to neutral chronological labels (v1, v2, ...) to avoid VLM bias

        # Save rollout images to output_dir for traceability
        if output_dir:
            rollout_images_dir = os.path.join(output_dir, "rollout_images")
            os.makedirs(rollout_images_dir, exist_ok=True)

        contents = []
        video_labels_str = ""
        for i, (video_key, data) in enumerate(results.items(), 1):
            label = f"v{i}"
            video_labels_str += f"  {label}  (Expected outcome: {data['tag'].upper()})\n"
            _, buffer = cv2.imencode('.png', cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR))
            image_bytes = buffer.tobytes()
            contents.append(types.Part.from_text(text=f"{label} (tag: {data['tag'].upper()}):"))
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
            print(f"  Added: {label} [{video_key}] (tag: {data['tag']})")

            # Save each mosaic to the run folder so we can audit exactly what was sent
            if output_dir:
                safe_key = video_key.replace(os.sep, "_")[:120]  # cap filename length
                mosaic_path = os.path.join(rollout_images_dir,
                                           f"{label}_{data['tag']}_{safe_key}.jpg")
                cv2.imwrite(mosaic_path, cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR))
                print(f"    saved → {os.path.relpath(mosaic_path, output_dir)}")

        # Add current observation — supports both PIL Image and cv2 numpy array (BGR)
        if isinstance(new_initial_state, np.ndarray):
            # cv2 array is BGR; encode directly to PNG bytes
            _, cur_obs_buf = cv2.imencode('.png', new_initial_state)
            current_obs_bytes = cur_obs_buf.tobytes()
        else:
            current_obs_buffer = io.BytesIO()
            new_initial_state.save(current_obs_buffer, format='PNG')
            current_obs_bytes = current_obs_buffer.getvalue()
        contents.append(types.Part.from_text(text="new_initial_state_image (predict for this environment):"))
        contents.append(types.Part.from_bytes(data=current_obs_bytes, mime_type="image/png"))

        # Add final prompt
        prompt = prompt_template.format(
            n_videos=total,
            video_labels=video_labels_str,
            task_instruction=task_instruction,
        )
        contents.append(types.Part.from_text(text=prompt))

        # Save the prompt to a text file for reproducibility
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            prompt_filepath = os.path.join(output_dir, "vlm_prompt.txt")
            with open(prompt_filepath, 'w') as _pf:
                _pf.write(prompt)
            print(f"  VLM prompt saved to: {prompt_filepath}")

        print(f"\nMaking single API call with {total} rollout images + 1 current observation...")

        try:
            response = vlm_client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )

            response_text = response.text
            print(f"\n  --- Raw VLM Response ---\n{response_text}\n  --- End of Response ---")

            # Try to extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text

            parsed_response = json.loads(json_str)

            vlm_responses = {
                'success': True,
                'n_rollouts': total,
                'video_keys': list(results.keys()),
                'analysis': parsed_response,
                'raw_response': response_text
            }

            if 'prediction_new_inital_state' in parsed_response:
                pred = parsed_response['prediction_new_inital_state']
                print(f"\n  Predicted outcome: {pred.get('predicted_outcome', 'N/A')}")
                print(f"  Recommended action: {pred.get('recommended_action', 'N/A')}")

            if 'prediction_rollouts' in parsed_response:
                rollouts = parsed_response['prediction_rollouts']
                descriptions = rollouts.get('initial_state_descriptions', [])
                frequencies = rollouts.get('success_frequency', [])
                print(f"  Initial states identified: {len(descriptions)}")
                for desc, freq in zip(descriptions, frequencies):
                    print(f"    - {str(desc)[:60]}... → success freq: {freq}")

        except json.JSONDecodeError as e:
            print(f"\n  Warning: JSON parse error — {e}")
            vlm_responses = {
                'success': True,
                'n_rollouts': total,
                'video_keys': list(results.keys()),
                'analysis': None,
                'raw_response': response_text,
                'parse_error': str(e)
            }
        except Exception as e:
            print(f"\n  Error during VLM call: {e}")
            vlm_responses = {
                'success': False,
                'n_rollouts': total,
                'video_keys': list(results.keys()),
                'error': str(e)
            }

        # Save responses to file
        if save_responses:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, output_file)
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base, ext = os.path.splitext(output_file)
                save_path = f"{base}_{timestamp}{ext}"

            with open(save_path, 'w') as f:
                json.dump(vlm_responses, f, indent=2)
            print(f"\n✓ VLM response saved to: {save_path}")
            
            # Add to response dict for reference
            vlm_responses['saved_to'] = save_path

        print(f"\n{'='*60}")
        print(f"VLM Analysis Complete — {'Success' if vlm_responses.get('success') else 'Failed'}")
        if output_dir:
            print(f"All files saved to: {output_dir}")
        print(f"{'='*60}\n")

        return vlm_responses


    def sam_vlm_planner(self, results, current_observation, front_cam_img, task_instruction, run_folder=None):
        # Call send_to_vlm with rollout results and current observation
        vlm_responses = self.send_to_vlm(
            results=results,
            prompt_template=prompt_template,
            task_instruction=task_instruction,
            new_initial_state=current_observation,
            output_dir=run_folder,
        )

        # Extract removable objects identified by the VLM
        parsed = (vlm_responses or {}).get("analysis") or {}
        inner = parsed.get("policy_execution_analysis") or {}
        deduced_actions = inner.get("deduced_actions") or []

        # Parse object names out of "@remove(object_name)" strings, ignoring "@add(...)" actions
        objects = []
        for action_str in deduced_actions:
            if isinstance(action_str, str):
                match = re.match(r'@(\w+)\((.+)\)', action_str)
                if match:
                    primitive = match.group(1).strip().lower()
                    if primitive == "add":
                        print(f"Ignoring unsupported 'add' action: {action_str}")
                        continue
                    objects.append(match.group(2).strip())

        data = {"object": objects}

        print(f"VLM planner response:{deduced_actions}")

        throw_points = []

        if len(objects)!= 0:
            for i in range(len(objects)):
                object_name = data["object"][i].replace("_", " ")

                pick_result = run_sam3_detection(
                    images=front_cam_img,
                    prompts=object_name,
                    output_dir=run_folder
                )

                pick_detections = pick_result[0]["detections"]
                if not pick_detections or len(pick_detections) == 0:
                    # raise ValueError(f"No detections found for object: {object_name}")
                    print(f"No detections found for object: {object_name}")
                    continue
                
                pick_bounding_boxes = pick_detections[0]['box_xyxy']
                u = (pick_bounding_boxes[0] + pick_bounding_boxes[2]) * 0.5
                v = (pick_bounding_boxes[1] + pick_bounding_boxes[3]) * 0.5
                u, v = int(round(u)), int(round(v))

                print(f"This is the pick result: {pick_result}")

                throw_points.append((u,v))
        
        return throw_points


if __name__=="__main__":
    success_path = "logs/stagecraft/stack_cups/success"
    failure_path = "logs/stagecraft/stack_cups/failure"

    # Load front and wrist images via PIL to avoid libpng/zlib conflicts with cv2
    front_img = cv2.cvtColor(np.array(Image.open("outputs/captured_images/front_img.png")), cv2.COLOR_RGB2BGR)
    wrist_img = cv2.cvtColor(np.array(Image.open("outputs/captured_images/wrist_img.png")), cv2.COLOR_RGB2BGR)
    front_img_rotated = cv2.rotate(front_img, cv2.ROTATE_180)
    new_initial_state = np.hstack([front_img_rotated, wrist_img])

    vlm_planner = SAM_VLM_Planner()

    vlm_context = vlm_planner.process_videos_with_tags(
        directories=[success_path, failure_path],
        performance_tags=['success', 'failure'],
    )

    throw_points = vlm_planner.sam_vlm_planner(
        results=vlm_context,
        current_observation=new_initial_state,
        task_instruction="stack the cups",
        front_cam_img=cv2.cvtColor(front_img_rotated, cv2.COLOR_BGR2RGB),
        run_folder="logs/stagecraft/stack_cups"
    )    
