import argparse
import gc
import os
import random
import time

import imageio
import torch
from diffusers.utils import load_image

from skyreels_v2_infer import DiffusionForcingPipeline
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines.image2video_pipeline import resizecrop
from moviepy.editor import VideoFileClip


def get_video_num_frames_moviepy(video_path):
    with VideoFileClip(video_path) as clip:
        num_frames = 0
        for _ in clip.iter_frames():
            num_frames += 1
        return clip.size, num_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="diffusion_forcing")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-DF-1.3B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"])
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--end_image", type=str, default=None)
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--ar_step", type=int, default=0)
    parser.add_argument("--causal_attention", action="store_true")
    parser.add_argument("--causal_block_size", type=int, default=1)
    parser.add_argument("--base_num_frames", type=int, default=97)
    parser.add_argument("--overlap_history", type=int, default=None)
    parser.add_argument("--addnoise_condition", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--use_usp", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A woman in a leather jacket and sunglasses riding a vintage motorcycle through a desert highway at sunset, her hair blowing wildly in the wind as the motorcycle kicks up dust, with the golden sun casting long shadows across the barren landscape.",
    )
    parser.add_argument("--prompt_enhancer", action="store_true")
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Higher speedup will cause to worse quality -- 0.1 for 2.0x speedup -- 0.2 for 3.0x speedup")
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        help="Using Retention Steps will result in faster generation speed and better generation quality.")
    args = parser.parse_args()

    args.model_id = download_model(args.model_id)
    print("model_id:", args.model_id)

    assert (args.use_usp and args.seed is not None) or (not args.use_usp), "usp mode need seed"
    if args.seed is None:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    if args.resolution == "540P":
        height = 544
        width = 960
    elif args.resolution == "720P":
        height = 720
        width = 1280
    else:
        raise ValueError(f"Invalid resolution: {args.resolution}")

    num_frames = args.num_frames
    fps = args.fps

    if num_frames > args.base_num_frames:
        assert (
            args.overlap_history is not None
        ), 'You are supposed to specify the "overlap_history" to support the long video generation. 17 and 37 are recommanded to set.'
    if args.addnoise_condition > 60:
        print(
            f'You have set "addnoise_condition" as {args.addnoise_condition}. The value is too large which can cause inconsistency in long video generation. The value is recommanded to set 20.'
        )

    guidance_scale = args.guidance_scale
    shift = args.shift
    
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)
    local_rank = 0
    if args.use_usp:
        assert not args.prompt_enhancer, "`--prompt_enhancer` is not allowed if using `--use_usp`. We recommend running the skyreels_v2_infer/pipelines/prompt_enhancer.py script first to generate enhanced prompt before enabling the `--use_usp` parameter."
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        import torch.distributed as dist

        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(dist.get_rank())
        device = "cuda"

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )

    prompt_input = args.prompt
    if args.prompt_enhancer and args.image is None:
        print(f"init prompt enhancer")
        prompt_enhancer = PromptEnhancer()
        prompt_input = prompt_enhancer(prompt_input)
        print(f"enhanced prompt: {prompt_input}")
        del prompt_enhancer
        gc.collect()
        torch.cuda.empty_cache()

    pipe = DiffusionForcingPipeline(
        args.model_id,
        dit_path=args.model_id,
        device=torch.device("cuda"),
        weight_dtype=torch.bfloat16,
        use_usp=args.use_usp,
        offload=args.offload,
    )

    if args.causal_attention:
        pipe.transformer.set_ar_attention(args.causal_block_size)
    
    if args.teacache:
        if args.ar_step > 0:
            num_steps = args.inference_steps + (((args.base_num_frames - 1) // 4 + 1) // args.causal_block_size - 1) * args.ar_step
            print('num_steps:', num_steps)
        else:
            num_steps = args.inference_steps
        pipe.transformer.initialize_teacache(enable_teacache=True, num_steps=num_steps, 
                                             teacache_thresh=args.teacache_thresh, use_ret_steps=args.use_ret_steps, 
                                             ckpt_dir=args.model_id)

    print(f"prompt:{prompt_input}")
    print(f"guidance_scale:{guidance_scale}")

    if os.path.exists(args.video_path):
        (v_width, v_height), input_num_frames = get_video_num_frames_moviepy(args.video_path)
        assert input_num_frames >= args.overlap_history, "The input video is too short."

        if v_height > v_width:
            width, height = height, width

        video_frames = pipe.extend_video(
            prompt=prompt_input,
            negative_prompt=negative_prompt,
            prefix_video_path=args.video_path,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=args.inference_steps,
            shift=shift,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            overlap_history=args.overlap_history,
            addnoise_condition=args.addnoise_condition,
            base_num_frames=args.base_num_frames,
            ar_step=args.ar_step,
            causal_block_size=args.causal_block_size,
            fps=fps,
        )[0]
    else:
        if args.image:
            args.image = load_image(args.image)
            image_width, image_height = args.image.size
            if image_height > image_width:
                height, width = width, height
            args.image = resizecrop(args.image, height, width)
            if args.end_image:
                args.end_image = load_image(args.end_image)
                args.end_image = resizecrop(args.end_image, height, width)

        image = args.image.convert("RGB") if args.image else None
        end_image = args.end_image.convert("RGB") if args.end_image else None
        
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(
                prompt=prompt_input,
                negative_prompt=negative_prompt,
                image=image,
                end_image=end_image,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=args.inference_steps,
                shift=shift,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                overlap_history=args.overlap_history,
                addnoise_condition=args.addnoise_condition,
                base_num_frames=args.base_num_frames,
                ar_step=args.ar_step,
                causal_block_size=args.causal_block_size,
                fps=fps,
            )[0]

    if local_rank == 0:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_out_file = f"{args.prompt[:100].replace('/','')}_{args.seed}_{current_time}.mp4"
        output_path = os.path.join(save_dir, video_out_file)
        imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
