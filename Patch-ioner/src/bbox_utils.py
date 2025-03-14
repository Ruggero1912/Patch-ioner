import torch
from copy import deepcopy
from PIL import ImageDraw
import itertools
import random


def extract_bboxes_feats(patch_embeddings, bboxes, gaussian_avg=False, 
                         gaussian_bbox_variance=0.5, get_single_embedding_per_image=False,
                         patch_size=14, attention_map=None):
    """
    if get_single_embedding_per_image is True, the weights of all the bounding boxes patches on an image will be summed and the function will return the patch weights depending on this map
    """
    N = patch_embeddings.shape[0]
    N_boxes = bboxes.shape[1]
    grid_size = int(patch_embeddings.shape[1]**0.5)
    device = patch_embeddings.device

    bboxes //= patch_size
    bboxes = bboxes.int()

    # Reshape patches to grid
    patch_embeddings = patch_embeddings.view(N, grid_size, grid_size, -1)  # Shape (N, grid_size, grid_size, embed_dim)
    if attention_map is not None:
        attention_map = attention_map.view(N, grid_size, grid_size)  # Shape (N, grid_size, grid_size)
    # Grid of the sum of the gaussian weights
    total_patch_weights = torch.zeros(N, grid_size, grid_size)

    # Extract boxes
    x1, y1, w, h = bboxes.unbind(-1)  # Separate box dimensions (N, N_boxes)

    # Create mesh grid for slicing
    x2 = x1 + w  # Exclusive end x
    y2 = y1 + h  # Exclusive end y

    means = []
    for i in range(N):
        image_means = []
        for j in range(N_boxes):
            if bboxes[i, j].sum().item() < 0 and get_single_embedding_per_image:
                # this is the case where we receive a dummy box
                continue
            # Extract the region for each box
            region_patches = patch_embeddings[i, y1[i, j]:y2[i, j] + 1, x1[i, j]:x2[i, j] + 1, :]  # (h, w, embed_dim)
            
            if attention_map is not None:
                patch_weights = attention_map[i, y1[i, j]:y2[i, j] + 1, x1[i, j]:x2[i, j] + 1]
                patch_weights /= patch_weights.sum()
                total_patch_weights[i, y1[i, j]:y2[i, j] + 1, x1[i, j]:x2[i, j] + 1] += patch_weights
                
                weighted_patches = region_patches * patch_weights.to(device).unsqueeze(-1)  # (h, w, embed_dim)
                region_mean = weighted_patches.sum(dim=(0, 1))  # Weighted mean
                
            elif gaussian_avg:
                # Create Gaussian weights
                h_span, w_span = region_patches.shape[:2]
                y_coords, x_coords = torch.meshgrid(
                    torch.linspace(-1, 1, h_span),
                    torch.linspace(-1, 1, w_span),
                    indexing="ij"
                )
                if gaussian_bbox_variance == 0:
                    patch_weights = torch.zeros((h_span, w_span))
                    # Determine central indices
                    center_y = [h_span // 2] if h_span % 2 == 1 else [h_span // 2 - 1, h_span // 2]
                    center_x = [w_span // 2] if w_span % 2 == 1 else [w_span // 2 - 1, w_span // 2]
                    # Randomly select one of the central elements in even case
                    center_y = random.choice(center_y)
                    center_x = random.choice(center_x)
                    # Set the selected central element to 1
                    patch_weights[center_y, center_x] = 1.0
                else:
                    distances = x_coords**2 + y_coords**2
                    patch_weights = torch.exp(-distances / gaussian_bbox_variance)
                    patch_weights = patch_weights / patch_weights.sum()  # Normalize to sum to 1

                # Apply Gaussian weights to region patches
                weighted_patches = region_patches * patch_weights.to(device).unsqueeze(-1)  # (h, w, embed_dim)
                region_mean = weighted_patches.sum(dim=(0, 1))  # Weighted mean
                
                # Recording the bbox weight inside the image patch weight map
                total_patch_weights[i, y1[i, j]:y2[i, j] + 1, x1[i, j]:x2[i, j] + 1] += patch_weights
            else:
                # Mean pooling case: create uniform weights
                h_span, w_span = region_patches.shape[:2]
                uniform_weights = torch.ones(h_span, w_span) / (h_span * w_span)
                
                # Update total_patch_weights for mean pooling
                total_patch_weights[i, y1[i,j]:y2[i,j]+1, x1[i,j]:x2[i,j]+1] += uniform_weights
                
                # Compute mean of the region
                region_mean = region_patches.mean(dim=(0, 1))

            # Store the mean
            image_means.append(region_mean)
        if not get_single_embedding_per_image:
            means.append(torch.stack(image_means))

    # Normalizing the weight map so the sum is equal to 1
    total_patch_weights /= total_patch_weights.sum(dim=(1,2), keepdim=True)
    if not get_single_embedding_per_image:
        return torch.stack(means)  # Shape (N, N_boxes, embed_dim)
    else:
        # Expand dimensions to match embeddings
        total_patch_weights = total_patch_weights.unsqueeze(-1).to(device)

        # Compute weighted sum
        weighted_patch_mean = (total_patch_weights * patch_embeddings).sum(dim=(1, 2))  
        return  weighted_patch_mean 
# Shape (N, embed_dim)

#def adjust_bbox_for_transform(image, bbox, resize_dim, crop_dim):
#    """
#    Adjusts the bounding box for a resized and center-cropped image.
#
#    Args:
#        image (PIL.Image): The input image.
#        bbox (list): The bounding box in [x1, y1, w, h] format.
#        resize_dim (int): The dimension of the shortest side after resizing.
#        crop_dim (int): The size of the square crop.
#
#    Returns:
#        list: The adjusted bounding box in [x1, y1, w, h] format.
#    """
#    x1, y1, w, h = bbox
#    orig_width, orig_height = image.size
#
#    # Calculate resize scale for the shortest side
#    if orig_width < orig_height:
#        scale = resize_dim / orig_width
#        resized_width, resized_height = resize_dim, int(orig_height * scale)
#    else:
#        scale = resize_dim / orig_height
#        resized_width, resized_height = int(orig_width * scale), resize_dim
#
#    # Scale the bounding box
#    x1 *= scale
#    y1 *= scale
#    w *= scale
#    h *= scale
#
#    # Calculate cropping offsets
#    crop_x = (resized_width - crop_dim) // 2
#    crop_y = (resized_height - crop_dim) // 2
#
#    # Adjust bounding box for cropping
#    x1 -= crop_x
#    y1 -= crop_y
#
#    # Clamp the bounding box to the cropped area
#    x1 = max(0, x1)
#    y1 = max(0, y1)
#    w = min(w, crop_dim - x1)
#    h = min(h, crop_dim - y1)
#
#    return [x1, y1, w, h]

def map_traces_to_grid(traces, n_patch):
    grid = torch.zeros((n_patch, n_patch))
    patch_size = 1.0 / n_patch
    
    for trace in traces:
        x, y = trace['x'], trace['y']
        if 0 <= x <= 1 and 0 <= y <= 1:
            grid_x, grid_y = int(x / patch_size), int(y / patch_size)
            grid[min(grid_y, n_patch - 1), min(grid_x, n_patch - 1)] += 1
    
    return grid

def adjust_bbox_for_transform(image, bbox, resize_dim, crop_dim):
    """
    Adjusts the bounding box for a resized and center-cropped image.

    Args:
        image (PIL.Image): The input image.
        bbox (list): The bounding box in [x1, y1, w, h] format.
        resize_dim (int): The dimension of the shortest side after resizing.
        crop_dim (int): The size of the square crop.

    Returns:
        list: The adjusted bounding box in [x1, y1, w, h] format.
    """
    x1, y1, w, h = bbox
    orig_width, orig_height = image.size

    # Scale factors for resizing
    if orig_width < orig_height:
        scale_w = resize_dim / orig_width
        scale_h = (resize_dim * orig_height) / orig_width / orig_height
    else:
        scale_h = resize_dim / orig_height
        scale_w = (resize_dim * orig_width) / orig_height / orig_width

    # New dimensions after resize
    new_width = int(orig_width * scale_w)
    new_height = int(orig_height * scale_h)

    # Update bounding box for resizing
    x1 = x1 * scale_w
    y1 = y1 * scale_h
    w = w * scale_w
    h = h * scale_h

    # Compute cropping offsets
    crop_x_offset = max(0, (new_width - crop_dim) // 2)
    crop_y_offset = max(0, (new_height - crop_dim) // 2)

    # Adjust bounding box for cropping
    x1 -= crop_x_offset
    y1 -= crop_y_offset

    # Clip bounding box to crop dimensions
    x1 = max(0, min(x1, crop_dim - 1))
    y1 = max(0, min(y1, crop_dim - 1))
    w = max(0, min(w, crop_dim - x1))
    h = max(0, min(h, crop_dim - y1))

    return [x1, y1, w, h]



def adjust_bbox_for_transform_no_scale(image, bbox, target_width, target_height):
    """
    - Does not preserve the image scale.
    Adjusts the bounding box for an image resized to a fixed width and height.

    Args:
        image (PIL.Image): The original image.
        bbox (list): The bounding box in [x1, y1, w, h] format.
        target_width (int): The width of the resized image.
        target_height (int): The height of the resized image.

    Returns:
        list: The adjusted bounding box in [x1, y1, w, h] format.
    """
    x1, y1, w, h = bbox
    orig_width, orig_height = image.size

    # Calculate scale factors for width and height
    scale_w = target_width / orig_width
    scale_h = target_height / orig_height

    # Adjust the bounding box
    x1 = x1 * scale_w
    y1 = y1 * scale_h
    w = w * scale_w
    h = h * scale_h

    # Return the adjusted bounding box
    return [x1, y1, w, h]


def draw_bounding_boxes(input_image, bounding_boxes, captions=[""], color="red", width=2, text_background=True, boxes_to_show = None):
    """
    Draws bounding boxes on an image.

    Args:
        image (PIL.Image): The image to draw on.
        bounding_boxes (list): A list of bounding boxes, each as [x1, y1, x2, y2].
        color (str): The color of the bounding boxes (default is red).
        width (int): The width of the bounding box lines (default is 2).

    Returns:
        PIL.Image: The image with bounding boxes drawn.
    """
    # Create a drawing context
    image = deepcopy(input_image)
    draw = ImageDraw.Draw( image )

    #scale = 720.0 / max(image.size)
    if boxes_to_show is not None:
        if isinstance(boxes_to_show, int):
            indexes_to_show = random.sample(range(len(bounding_boxes)), boxes_to_show)
        else:
            indexes_to_show = boxes_to_show
    
    for i, (bbox, cap ) in enumerate(itertools.zip_longest(bounding_boxes, captions, fillvalue="")):

        if boxes_to_show is not None:
            if i not in indexes_to_show: continue
        #bbox = [ i / scale for i in bbox ]
        #x1, y1, w, h = bbox
        x1, y1, x2, y2 = bbox
        
        #x2, y2 = x1 + w, y1 + h  # Convert width/height to bottom-right corner
        try:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            if cap != "":
                if text_background:
                    left,top,right,bottom = draw.multiline_textbbox((x1,y1), cap) #textbbox
                    draw.rectangle((left-5, top-5, right+5, bottom+5), fill="white")
                draw.multiline_text((x1,y1), cap, fill=color)   #text
            
        except Exception as e:
            print("exception, i: ", i, f"{x1 = } {y1 = } {x2 = }, {y2 = }")
            print(e)
    
    return image

def extract_bboxes_feats_double_dino(dino_model, patch_embeddings, bboxes, cls_token, registers_tokens, patch_size, return_type="cls", gaussian_bbox_variance=0.5):
    """
    Perform a forward pass of the last DINO layer with selected features, batched.

    Args:
        dino_model: The DINO model.
        patch_embeddings: Patch embeddings before the last layer.
        bboxes: Bounding boxes for each image in the batch (BS x N_BOX_MAX x 4).
        cls_token: CLS token embedding.
        return_type: Type of feature to return ('cls', 'avg', 'gaussian_avg').
        gaussian_bbox_variance: Variance for Gaussian averaging.

    Returns:
        bbox_features: Features for each bounding box based on return_type.
    """
    N = patch_embeddings.shape[0]  # Batch size
    N_boxes = bboxes.shape[1]      # Number of bounding boxes
    grid_size = int(patch_embeddings.shape[1] ** 0.5)  # Assuming square grid
    embed_dim = patch_embeddings.shape[-1]

    bboxes_patch_indexes = bboxes.clone()
    bboxes_patch_indexes //= patch_size  # Scale down bbox coordinates to match patch grid
    bboxes_patch_indexes = bboxes_patch_indexes.int()

    # Reshape patches to grid
    patch_embeddings = patch_embeddings.view(N, grid_size, grid_size, embed_dim)  # (N, grid_size, grid_size, embed_dim)

    if cls_token is not None:
        cls_tokens = cls_token.view(N, embed_dim)
        if registers_tokens is not None:
            patches_offset = 5
        else:
            patches_offset = 1
    else:
        assert return_type != "cls"
        patches_offset = 0
    batch_outputs = []

    #batch_inputs = []

    means = []
    for i in range(N):  # Iterate over batch
        image_means = []
        
        if cls_token is not None:
            cls_cur_img = cls_tokens[i].reshape(1, 1, embed_dim)
            if registers_tokens is not None:
                cur_img_register_tokens = registers_tokens[i].reshape(1, 4, embed_dim)
            
        for j in range(N_boxes):  # Iterate over bounding boxes
            # Extract the region for the bounding box
            region_patches_xy = patch_embeddings[i, bboxes_patch_indexes[i, j, 1]:bboxes_patch_indexes[i, j, 3] + 1, bboxes_patch_indexes[i, j, 0]:bboxes_patch_indexes[i, j, 2] + 1, :]
            #region_patches = region_patches.reshape(-1, embed_dim)  # Flatten to (num_patches, embed_dim)

            #region_patches = region_patches.view(-1, embed_dim)  # Flatten to (num_patches, embed_dim)
            #cls_cur_img = cls_tokens[i].unsqueeze(0)  # Add batch dimension (1, embed_dim)
            #region_patches = region_patches.unsqueeze(0)  # Add batch dimension (1, num_patches, embed_dim)
            region_patches = region_patches_xy.reshape(1,-1, embed_dim)
            if cls_token is not None:
                inputs = torch.cat([cls_cur_img, region_patches], dim=1)  # Concatenate along the token dimension (1, num_patches + 1, embed_dim)
                if registers_tokens is not None:
                    inputs = torch.cat([cls_cur_img, cur_img_register_tokens, region_patches], dim=1)  # Concatenate along the token dimension (1, num_patches + 5, embed_dim)
            else:
                inputs = torch.cat([region_patches], dim=1)  # Concatenate along the token dimension (1, num_patches + 1, embed_dim)
                
            outputs = dino_model.blocks[-1](inputs)  # Forward pass
            # shape (1, 1 + len(region_patches), 768)
            #cls_cur_img = cls_tokens[i]
            #cls_cur_img = cls_cur_img.reshape(1, embed_dim)
            #inputs = torch.cat([cls_cur_img, region_patches], dim=0)  # Add CLS token to inputs
            #outputs = dino_model.blocks[-1](inputs)  # Forward pass
            
            batch_outputs.append(outputs)

            region_patches = outputs[0, patches_offset: ,] #(1,45,768) -> (1,1,768)
            
            if return_type == "gaussian_avg":
                #region_patches = outputs[5: ,]
                h_span, w_span = region_patches_xy.shape[:2]
                y_coords, x_coords = torch.meshgrid(
                    torch.linspace(-1, 1, h_span),
                    torch.linspace(-1, 1, w_span),
                    indexing="ij"
                )
                distances = x_coords**2 + y_coords**2
                gaussian_weights = torch.exp(-distances / gaussian_bbox_variance)  # Adjust 0.1 for variance control
                gaussian_weights = gaussian_weights / gaussian_weights.sum()  # Normalize to sum to 1

                # Apply Gaussian weights to region patches
                weighted_patches = region_patches_xy * gaussian_weights.to(next(dino_model.parameters()).device).unsqueeze(-1)  # (h, w, embed_dim)
                region_mean = weighted_patches.sum(dim=(0,1))  # Weighted mean
                #image_means.append(region_mean)
            elif return_type == "avg":
                # Compute mean of the region
                region_mean = region_patches.mean(dim=(0))  # Mean over h, w
            elif return_type == "cls":
                region_mean = outputs[0, 0, ]
            image_means.append(region_mean)

        means.append(torch.stack(image_means))

    stacked_means = torch.stack(means)
    #stacked_means = stacked_means.reshape(-1, embed_dim)
    return stacked_means


def process_bboxes(imgs, bboxes, transform):
    transformed_bboxes = []
    bboxes = bboxes.tolist()
    for img, img_bboxes in zip(imgs, bboxes):
        for bbox in img_bboxes:
            # Crop the region defined by bbox
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h
            cropped_region = img.crop((x_min, y_min, x_max, y_max))
            
            # Apply the transform to the cropped region
            transformed_region = transform(cropped_region)
            transformed_bboxes.append(transformed_region)
    
    return torch.stack(transformed_bboxes)