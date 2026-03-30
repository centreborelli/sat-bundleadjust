from bundle_adjust import loader
import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image

# helper to make arrays PNG-writable via PIL
def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)
        return (255 * arr).clip(0, 255).astype(np.uint8)

    return np.clip(arr, 0, 255).astype(np.uint8)

def compute_SSIM(array1, array2):
    assert array1.shape[0] == array2.shape[0] and array1.shape[1] == array2.shape[1]
    ssim_obj = cv2.quality.QualitySSIM.create(array1)
    raw_score = ssim_obj.compute(array2)[0] 
    score = max(0.0, min(1.0, raw_score)) # SSIM range is [-1, 1] -> clip to [0, 1] because SSIM < 0 is too low anyway
    return score

def prepare_image_collection_for_SSIM(images, output_size):
    # SSIM is faster to compute on downsampled images of the same size
    # this function receives a list of SatelliteImage and outputs a list of corresponding downsampled numpy arrays with the same size
    ds_images = []
    h, w = output_size, output_size
    for idx, image in enumerate(images):
        ds_image = loader.load_image(image.geotiff_path, offset=image.offset)
        if idx == 0 and output_size is None:
            h, w = ds_image.shape[0], ds_image.shape[1]
        ds_image = cv2.resize(ds_image, (w, h), interpolation=cv2.INTER_AREA)
        ds_images.append(ds_image)
    return ds_images

def classify_challenging_pairs_SSIM(pairs_to_match, images, SSIM_threshold=0.2, size_for_SSIM=256):

    print("\nSearching for challenging pairs with SSIM...")
    print(f"SSIM_threshold = {SSIM_threshold:.2f}\n")

    challenging_pairs = []
    scores_list = []
    challenging_pairs = []

    # (1) downsample because images need to be the same size and SSIM computation will be faster
    ds_images = prepare_image_collection_for_SSIM(images, size_for_SSIM)

    # (2) compute SSIM via opencv implementation
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair[0], pair[1]
        score = compute_SSIM(ds_images[i], ds_images[j])
        scores_list.append(score)
        if score <= SSIM_threshold:
            challenging_pairs.append((i, j))

        DEBUG=False # set this to True to store all pairs sorted by SSIM in a hardcoded directory
        if DEBUG:
            debug_dir = "deleteme_debug_auto_ssim"
            os.makedirs(debug_dir, exist_ok=True)
            # match dtype
            img_i, img_j = _to_uint8(ds_images[i]), _to_uint8(ds_images[j])
            concat_img = np.concatenate([img_i, img_j], axis=1)
            image_i_id = loader.get_id(images[i].geotiff_path)
            image_j_id = loader.get_id(images[j].geotiff_path)
            out_path = f"{debug_dir}/{score:.4f}_{image_i_id}_{image_j_id}.png"
            Image.fromarray(concat_img).save(out_path)

    return challenging_pairs













#### DINO code starts here

def _open_rgb_image(image_or_path):
    """
    Accepts either:
      - a SatelliteImage-like object with .geotiff_path
      - a string path
      - a numpy array image

    Returns
    -------
    PIL.Image.Image
        RGB PIL image.
    """
    if hasattr(image_or_path, "geotiff_path"):
        path = image_or_path.geotiff_path
        return Image.open(path).convert("RGB")

    if isinstance(image_or_path, np.ndarray):
        arr = image_or_path

        # handle grayscale
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")

        # handle H x W x C
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                return Image.fromarray(arr.squeeze(axis=2)).convert("RGB")
            if arr.shape[2] in (3, 4):
                return Image.fromarray(arr).convert("RGB")

        raise ValueError(f"Unsupported numpy image shape: {arr.shape}")

    return Image.open(image_or_path).convert("RGB")

def load_dino_model(
    model_name="facebook/dinov3-vitl16-pretrain-sat493m",
    device=None,
    torch_dtype=None,
):
    """
    Loads the DINOv3 processor + model.

    Notes:
    - This model is gated on Hugging Face, so you need access/auth configured.
    - DINOv3 support requires a recent Transformers version.
    """
    from transformers import AutoImageProcessor, AutoModel
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        if device == "cuda":
            # bfloat16 is often a good default on recent GPUs
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
    token = os.environ.get("HF_TOKEN", None)
    processor = AutoImageProcessor.from_pretrained(model_name, token=token)
    model = AutoModel.from_pretrained(model_name, token=token, torch_dtype=torch_dtype)    
    model.to(device)
    model.eval()

    return processor, model, device

def compute_dino_embedding(
    image_or_path,
    processor,
    model,
    device,
    normalize=True,
):
    """
    Returns one global embedding vector for the image.

    We use the CLS token (token 0) from last_hidden_state as the global image embedding.
    """
    import torch
    import torch.nn.functional as F

    image = _open_rgb_image(image_or_path)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # DINOv3 ViT returns token embeddings in last_hidden_state:
    # [batch, num_tokens, hidden_dim]
    # token 0 is the class token
    with torch.inference_mode():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]   # shape: [1, hidden_dim]
    embedding = embedding.squeeze(0)

    # convert to float32 for stable similarity math / CSV friendliness later
    embedding = embedding.float()

    if normalize:
        embedding = F.normalize(embedding, dim=0)

    return embedding.cpu().numpy()

def prepare_image_collection_for_DINO(images, processor, model, device):
    """
    Compute one DINO embedding per image, once.

    Returns
    -------
    embeddings : list[np.ndarray]
        L2-normalized embedding for each image.
    """
    embeddings = []
    for img in tqdm(images, desc="Computing DINO embeddings"):
        emb = compute_dino_embedding(
            img,
            processor=processor,
            model=model,
            device=device,
            normalize=True,
        )
        embeddings.append(emb)
    return embeddings

def cosine_similarity_np(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def classify_challenging_pairs_DINO(
    pairs_to_match,
    images,
    cosine_similarity_threshold=0.8,
    model_name="facebook/dinov2-large",
    device=None,
):
    """
    Classify challenging pairs using DINO embeddings + cosine similarity.

    A pair is considered challenging if:
        cosine_similarity < cosine_similarity_threshold

    Parameters
    ----------
    pairs_to_match : list[tuple[int, int]]
        List of image index pairs.
    images : list
        Collection of images / SatelliteImage objects.
    cosine_similarity_threshold : float
        Pairs below this similarity are marked as challenging.
    model_name : str
        Hugging Face model name.
    device : str or None
        "cuda", "cpu", or None for auto.

    Returns
    -------
    challenging_pairs : list[tuple[int, int]]
    """
    print("\nSearching for challenging pairs with DINO...")
    print(f"cosine_similarity_threshold = {cosine_similarity_threshold:.2f}\n")

    challenging_pairs = []
    scores_list = []

    # (1) compute embeddings once
    processor, model, device = load_dino_model(
        model_name=model_name,
        device=device,
    )
    ds_images = prepare_image_collection_for_SSIM(images, 256)
    embeddings = prepare_image_collection_for_DINO(ds_images, processor, model, device)
    

    # (2) compare pairs via cosine similarity
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair[0], pair[1]

        score = cosine_similarity_np(embeddings[i], embeddings[j])
        scores_list.append(score)

        if score < cosine_similarity_threshold:
            challenging_pairs.append((i, j))

        DEBUG = False  # set True to store all pairs sorted by cosine similarity
        if DEBUG:
            debug_dir = "deleteme_debug_auto_dino"
            os.makedirs(debug_dir, exist_ok=True)
            # match dtype
            img_i, img_j = _to_uint8(ds_images[i]), _to_uint8(ds_images[j])
            concat_img = np.concatenate([img_i, img_j], axis=1)
            image_i_id = loader.get_id(images[i].geotiff_path)
            image_j_id = loader.get_id(images[j].geotiff_path)
            out_path = f"{debug_dir}/{score:.4f}_{image_i_id}_{image_j_id}.png"
            Image.fromarray(concat_img).save(out_path)
    return challenging_pairs
