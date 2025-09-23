import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader



def eval_model11(args):
    disable_torch_init()

    # Depth Model
    depth_model, depth_transform = get_depth_predictor()
    print("Depth model successfully loaded!")

    # Model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)


    with open(args.annotation_file) as f:
        questions = json.load(f)

    #questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    #answers_file = os.path.expanduser(args.answers_file)
    #os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    #ans_file = open(answers_file, "w")
    
    mask_processer = copy.deepcopy(image_processor)
    mask_processer.do_normalize = False
    mask_processer.do_convert_rgb = False
    mask_processer.rescale_factor = 1.0

    questions = [questions] #This is additional code by me 
    
    for line in tqdm(questions, total=len(questions)):
        #question_id = line["id"]
        #image_file = line["image_info"]["file_path"]
        #image_info = line["image_info"]
        #text_question = line["text_q"]
        #qa_info = line["qa_info"]
        # generate mask

        image = Image.open('/home/louyujing/datasets/[target_dir/train]/'+line['filename']+'.jpg').convert("RGB")
        line['bbox'] =line['bbox']


        image_cv = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        image_with_bbox =draw_bbox(image_cv,line['bbox'])
        cv2.imwrite('./show.png',image_with_bbox)
        #exit()
        image_info ={}
        image_info["height"], image_info["width"] = image.size[0], image.size[1]
        if args.use_mask:

            masks = []
            try:
                rles = line["rle"]
                for rle in rles:
                    m = cocomask.decode(rle)
                    m = m.astype(np.uint8)
                    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
                    if image_aspect_ratio == "pad":
                        m = pad_to_square(m)
                    masks.append(m)

            except:
                bboxes = line["bbox"]
                for bbox in bboxes:
                    zero_mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                    clamp_box(bbox, image_info)
                    x1, y1, x2, y2 = map(int, bbox)
                    zero_mask[y1:y2, x1:x2] = 1
                    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
                    if image_aspect_ratio == "pad":
                        zero_mask = pad_to_square(zero_mask)
                    masks.append(zero_mask)

        else:
            masks = []
            print("using box!")
            bboxes = line["bbox"]

            for bbox in bboxes:
                zero_mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                clamp_box(bbox, image_info)
                x1, y1, x2, y2 = map(int, bbox)
                zero_mask[y1:y2, x1:x2] = 1
                image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
                if image_aspect_ratio == "pad":
                    zero_mask = pad_to_square(zero_mask)
                masks.append(zero_mask)

        if len(masks) > 0:
            masks_pt = []
            for m in masks:
                m = mask_processer.preprocess(m[None, ...], return_tensors="pt")["pixel_values"][0]
                masks_pt.append(m)
            masks = torch.vstack(masks_pt).float()  # (n, h, w)
        else:
            masks = None


        with torch.no_grad():


            depth_input_image = np.array(image)
            colorized_depth = get_depth_map(depth_input_image, depth_model, depth_transform)

            images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
            depths_tensor = process_images([Image.fromarray(colorized_depth)], image_processor, model.config).to(
                model.device, dtype=torch.float16
            )

            conv = conv_templates[args.conv_mode].copy()
            
            conversations = line["conversations"]

            num_question = len(conversations) // 2
            for i in range(num_question):
                question = conversations[i * 2]["value"]

                #question, _ = re.subn(r"<mask>", "<mask> <depth>", question)
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

            #prompt = conv.get_prompt()+ 'How large is the whilte car in the middle of the road ?'
            #prompt = '<image>\nYou are a photographer taking a picture of two people with \
            #faces in <mask> <depth> and <mask> <depth>. Can you infer the age of the first person?'
            prompt = '<image>\n What is the distance between  <mask> <depth> and <mask> <depth> ?'
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids = input_ids.to(device="cuda", non_blocking=True)
            input_ids = input_ids.unsqueeze(0)

            stop_str = (
                conv_templates[args.conv_mode].sep
                if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
                else conv_templates[args.conv_mode].sep2
            )

            model.to(dtype=torch.bfloat16)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor.to(dtype=torch.bfloat16, device="cuda", non_blocking=True),
                    depths=depths_tensor.to(dtype=torch.bfloat16, device="cuda", non_blocking=True),
                    masks=[masks.to(dtype=torch.bfloat16, device="cuda", non_blocking=True)]
                    if masks is not None
                    else None,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=128,
                    use_cache=True,
                )

            outputs = outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            print(outputs)
            #import pdb
            #pdb.set_trace()
            exit()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "image": image_file,
                        "question": text_question,
                        "pred": outputs,
                        "gt": conversations[i * 2 + 1]["value"],
                        "model_id": model_name,
                        "qa_info": qa_info,
                    }
                )
                + "\n"
            )

    ans_file.close()



def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    if 'VILA' in model_path:
        path = '/mnt/tqnas/ali/home/lubin.fan/workspace2/SpatialRGPT-main/spatialrgpt.pt'
        models = torch.load(path)
        tokenizer, model, image_processor, context_len = models['tokenizer'], models['model'], models['image_processor'], models['context_len']

    else:
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    model.use_depth =False
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        image = Image.open(os.path.join(args.image_folder, line['image'])).convert('RGB')

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        model.gt_depth = False

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                ori_imgs=[image],
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
