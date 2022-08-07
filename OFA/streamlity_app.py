import streamlit as st 
from PIL import Image
import io
from googletrans import Translator
from gtts import gTTS
import torch
import numpy as np
from models.ofa import OFAModel
from PIL import Image
from fairseq import utils,tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from torchvision import transforms

#############################################################################################
@st.cache(suppress_st_warning=True , allow_output_mutation=True)
def load_model():
    # Register caption task
    tasks.register_task('caption',CaptionTask)

    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = False


    # Load pretrained ckpt & config
    overrides={"bpe_dir":"utils/BPE", "eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths('checkpoints/caption.pt'),
            arg_overrides=overrides
        )

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    return task, generator, models, cfg, use_cuda, use_fp16


def transform_image(cfg,task):
    # Image transform
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return patch_resize_transform


def encode_text(text, task, length=None, append_bos=False, append_eos=False):
    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    # task
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    

    return s



# Construct input for caption task
def construct_sample(patch_resize_transform, task, image: Image):
    # patch_resize_transform , encode_text --> task
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", task,append_bos=True, append_eos=True).unsqueeze(0)
    # text preprocess
    pad_idx = task.src_dict.pad()
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t




########################################################################################################

def get_Caption(image, use_cuda, use_fp16, task, generator, models, patch_resize_transform):
    # use_cuda , use_fp16 , task, generator, models
    # Construct input sample & preprocess for GPU if cuda available
    sample = construct_sample(patch_resize_transform, task, image)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    # Run eval step for caption
    with torch.no_grad():
        result, scores = eval_step(task, generator, models, sample)

    caption = result[0]['caption']
    return caption


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
#         st.image('/home/ubuntu/content/test/test_img.png')
#         return Image.open('/home/ubuntu/content/test/test_img.png')


def text_to_speech(caption):   
    audio_english = gTTS(text=caption, lang='en', slow=False)
    en_file_name = "enAudio" 
    audio_english.save(f"{en_file_name}.mp3")
    translator = Translator()
    #translate text
    translated_text = translator.translate(caption, src='en', dest = 'ar').text
    
    audio_arabic = gTTS(text=translated_text, lang='ar', slow=False)
    ar_file_name = "arAudio"
    audio_arabic.save(f"{ar_file_name}.mp3")
    return translated_text, en_file_name, ar_file_name
    


def main():
    st.title('Image Caption Generator')
    #image_path = '/home/ubuntu/content/test/test_img.png'
    task, generator, models, cfg, use_cuda, use_fp16 = load_model()
    patch_resize_transform = transform_image(cfg,task)
    image_path = load_image()
    if (image_path != None) : 
        caption = get_Caption(image_path, use_cuda, use_fp16, task, generator,
                              models, patch_resize_transform)

        result_btn = st.button('Generate Caption & Audio')
        if result_btn:
            st.write('Generating Caption...')
            st.write(caption)

            st.write('Generating Audio...')
            ar_text, en_file_name, ar_file_name = text_to_speech(caption)

            #english
            audio_file_en = open(f"{en_file_name}.mp3", "rb")
            audio_bytes_en = audio_file_en.read()
            st.markdown(f"## English audio:")
            st.audio(audio_bytes_en, format="audio/mp3", start_time=0)

            st.write('Arabic Translated Caption:')
            st.write(ar_text)
            #arabic
            audio_file_ar = open(f"{ar_file_name}.mp3", "rb")
            audio_bytes_ar = audio_file_ar.read()
            st.markdown(f"## Arabic audio:")
            st.audio(audio_bytes_ar, format="audio/mp3", start_time=0)

if __name__ == '__main__':
    main()