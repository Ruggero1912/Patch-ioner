import pynvml
import torch

def get_gpu_with_most_memory():
    """
    Return the CUDA device with the most free memory, as a string like 'cuda:0'.
    Uses NVML for accurate memory reporting.
    """
    pynvml.nvmlInit()
    best_gpu = None
    max_free_mem = 0

    for i in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem_info.free > max_free_mem:
            best_gpu = i
            max_free_mem = mem_info.free

    pynvml.nvmlShutdown()
    if best_gpu is not None:
        return f"cuda:{best_gpu}"
    else:
        raise RuntimeError("No CUDA devices available")

import pandas as pd

def get_model_infos(model_name):
    """
    - returns the model name, image size, number of patches, and backbone name
    """
    model_name = model_name.replace(".karpathy", "").replace(".k", "")
    models_dict = {
        "viecap_b16_14patches" : ("ViECap@224", 14, "CLIP B16"),
        "meacap_invlm_b16_14patches" : ("MeaCap@224", 14, "CLIP B16"),
        "viecap" : ("ViECap@224", 7, "CLIP B32"),
        "meacap_invlm" : ("MeaCap@224", 7, "CLIP B32"),
        "viecap_b16_37patches" : ("ViECap@592", 37, "CLIP B16"),
        "meacap_invlm_b16_37patches" : ("MeaCap@592", 37, "CLIP B16"),
        "openclip_H14_noise_0_016" : ("Noise@224 0.016", 7, "OpenCLIP H14"),
        "openclip_H14_noise_0_14_epoch20" : ("Noise@224 0.14 Epoch 20", 7, "OpenCLIP H14"),
        "openclip_H14_mix_noise_0_04_epoch20" : ("Mix-Noise@224 0.04 Epoch 20", 7, "OpenCLIP H14"),
        "original_decap" : ("DeCap@224", 7, "CLIP B32"),
        "original_decap_B16" : ("DeCap@224", 14, "CLIP B16"),
        "original_decap_big_resize_B16" : ("DeCap@592", 37, "CLIP B16"),
        "INViTE_B16_1layer" : ("DeCap@224", 14, "INViTE B16 1 Layer"),
        "INViTE_B16_2layer" : ("DeCap@224", 14, "INViTE B16 2 Layers"),
        "INViTE_B16_3layer" : ("DeCap@224", 14, "INViTE B16 3 Layers"),
        "INViTE_B32_1layer" : ("DeCap@224", 7, "INViTE B32 1 Layer"),
        "INViTE_B32_2layer" : ("DeCap@224", 7, "INViTE B32 2 Layers"),
        "INViTE_B32_3layer" : ("DeCap@224", 7, "INViTE B32 3 Layers"),
        "INViTE_B16_1layer_bigResize" : ("DeCap@592", 37, "INViTE B16 1 Layer"),
        "INViTE_B16_2layer_bigResize" : ("DeCap@592", 37, "INViTE B16 2 Layers"),
        "INViTE_B16_3layer_bigResize" : ("DeCap@592", 37, "INViTE B16 3 Layers"),
        "regionclip_resnet50x4_p32" : ("DeCap@288",9, "RegionCLIP ResNet50x4"),
        "regionclip_resnet50_p32" : ("DeCap@224", 7, "RegionCLIP ResNet50"),
        "dinotxt" : ("Patchioner@518", 37, "DINOv2 B14 DINO.txt"),
        "mlp" : ("Patchioner@518", 37, "DINOv2 B14 T2D"),
        "mlp.viecap" : ("ViECap@518", 37, "DINOv2 B14 T2D"),
        "mlp.meacap" : ("MeaCap@518", 37, "DINOv2 B14 T2D"),
        "mlp_noise" : ("Patchioner-Noise@518", 37, "DINOv2 B14 T2D"),
        "mlp_noproj" : ("Patchioner-NoProj@518", 37, "DINOv2 B14 T2D"),
        "proxyclip_b16_dinov2" : ("DeCap@518", 14, "ProxyCLIP B16 DINOv2 B14"),
        "proxyclip_b16" : ("DeCap@296", 8, "ProxyCLIP B16 DINOv1 B8"),
        "denseclip_B16_seg_40patches" : ("DeCap@640", 40, "DenseCLIP B16"),
        "alphaclip_B16" : ("DeCap@224", 7, "AlphaCLIP B16"),
        "alphaclip_B16_CLS" : ("DeCap@224", 7, "AlphaCLIP B16 CLS"),
        "clipcap_dino_vitb14" : ("ClipCap@518", 37, "DINOv2 B14"),
        "clipcap_dino_vitb14_trf" : ("ClipCap-Trf@518", 37, "DINOv2 B14"),
        "clipcap_dino_vitb14_trfGPT" : ("ClipCap-TrfGPT@518", 37, "DINOv2 B14"),
        "clipcap_dino_vitb14_GPT" : ("ClipCap-GPT@518", 37, "DINOv2 B14"),
        "clipcap_clip_vitb32.paper" : ("ClipCap.orig@224", 7, "CLIP B32"),
        "clipcap_dino_vitb14_avgpatch_trfGPT" : ("ClipCap-AvgPatchTrfGPT@518", 37, "DINOv2 B14"),
        "clipcap_dino_vitb14_avgpatch_GPT" : ("ClipCap-AvgPatchGPT@518", 37, "DINOv2 B14"),
        "clipcap_dino_vitb14_avgpatch" : ("ClipCap-AvgPatch@518", 37, "DINOv2 B14"),
        "clipcap_dino_vitb14_avgpatch_trf" : ("ClipCap-AvgPatchTrf@518", 37, "DINOv2 B14"),
        
        "clipcap_dino_vitb14_attn_trfGPT" : ("ClipCap-AttnTrfGPT@518", 37, "DINOv2 B14"),
        "clipcap_dino_vitb14_attn_GPT" : ("ClipCap-AttnGPT@518", 37, "DINOv2 B14"),
        "clipcap_dino_vitb14_attn" : ("ClipCap-Attn@518", 37, "DINOv2 B14"),
        "clipcap_dino_vitb14_attn_trf" : ("ClipCap-AttnTrf@518", 37, "DINOv2 B14"),

        "clipcap_dino_vitb14_patch_most_attended_GPT" : ("ClipCap-Patch-MostAttendedGPT@518", 37, "DINOv2 B14"),
        "clipcap_dino_vitb14_patch_near_CLS_GPT" : ("ClipCap-Patch-NearCLS-GPT@518", 37, "DINOv2 B14"),
        "clipcap_dino_vitb14_patch_near_capt_t2d_GPT" : ("ClipCap-Patch-NearCaption-T2D-GPT@518", 37, "DINOv2 B14"),

        "clipcap_dino_vitl14_patch_near_capt_DINOtxt_GPT" : ("ClipCap-Patch-NearCaption-DINOtxt-GPT@518", 37, "DINOv2 L14"),

    }
    return models_dict[model_name]


def format_dataframe_with_std(df: pd.DataFrame, num_digits: int = 1, is_dense_capt=False, show_stddev: bool = True):
    """
    Format a dataframe by combining score columns with their standard deviations.
    Similar to print_latex_table but returns a modified dataframe instead of printing.
    
    Args:
        df: Input dataframe with score columns and their std counterparts
        num_digits: Number of decimal places for scores (except inference time which uses 3)
        is_dense_capt: Whether this is dense captioning (affects column order)
    
    Returns:
        Modified dataframe with formatted score columns
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Helper function to format values with standard deviation
    def format_with_std(value, std_value, digits):
        try:
            if pd.isna(value) or value == '':
                return ""
            val_float = float(value)
            if not show_stddev or (pd.isna(std_value) or std_value == ''):
                return f"{val_float:.{digits}f}"
            std_float = float(std_value)
            return f"{val_float:.{digits}f}±{std_float:.{digits}f}"
        except (ValueError, TypeError):
            return f"{value}" if not pd.isna(value) and value != '' else ""
    
    # Apply formatting to each row
    for idx, row in result_df.iterrows():
        # Determine digits for inference time (always 3)
        time_digits = 3
        
        # Format score columns with their standard deviations
        if 'Bleu_4' in result_df.columns:
            result_df.at[idx, 'Bleu_4'] = format_with_std(
                row.get('Bleu_4', ''), row.get('Bleu_4_std', ''), num_digits)
        
        if 'METEOR' in result_df.columns:
            result_df.at[idx, 'METEOR'] = format_with_std(
                row.get('METEOR', ''), row.get('METEOR_std', ''), num_digits)
        
        if 'ROUGE_L' in result_df.columns:
            result_df.at[idx, 'ROUGE_L'] = format_with_std(
                row.get('ROUGE_L', ''), row.get('ROUGE_L_std', ''), num_digits)
        
        if 'CIDEr' in result_df.columns:
            result_df.at[idx, 'CIDEr'] = format_with_std(
                row.get('CIDEr', ''), row.get('CIDEr_std', ''), num_digits)
        
        if 'SPICE' in result_df.columns:
            result_df.at[idx, 'SPICE'] = format_with_std(
                row.get('SPICE', ''), row.get('SPICE_std', ''), num_digits)
        
        if 'RefPAC-S' in result_df.columns:
            result_df.at[idx, 'RefPAC-S'] = format_with_std(
                row.get('RefPAC-S', ''), row.get('RefPAC-S_std', ''), num_digits)
        
        if 'CLIP-S' in result_df.columns:
            result_df.at[idx, 'CLIP-S'] = format_with_std(
                row.get('CLIP-S', ''), row.get('CLIP-S_std', ''), num_digits)
        
        if 'PAC-S' in result_df.columns:
            result_df.at[idx, 'PAC-S'] = format_with_std(
                row.get('PAC-S', ''), row.get('PAC-S_std', ''), num_digits)

        if 'CLIP-S_cropped' in result_df.columns:
            result_df.at[idx, 'CLIP-S_cropped'] = format_with_std(
                row.get('CLIP-S_cropped', ''), row.get('CLIP-S_cropped_std', ''), num_digits)

        if 'PAC-S_cropped' in result_df.columns:
            result_df.at[idx, 'PAC-S_cropped'] = format_with_std(
                row.get('PAC-S_cropped', ''), row.get('PAC-S_cropped_std', ''), num_digits)

        # Format inference time (always uses 3 digits)
        if 'avg_inference_time_per_image' in result_df.columns:
            result_df.at[idx, 'avg_inference_time_per_image'] = format_with_std(
                row.get('avg_inference_time_per_image', ''), 
                row.get('std_inference_time_per_image', ''), time_digits)
        
        # Format map_score for dense captioning (always 2 digits)
        if is_dense_capt and 'map_score' in result_df.columns:
            try:
                if not pd.isna(row.get('map_score', '')) and row.get('map_score', '') != '':
                    result_df.at[idx, 'map_score'] = f"{float(row['map_score']):.2f}"
            except (ValueError, TypeError):
                pass
    
    # Remove the std columns since they're now incorporated into the main columns
    std_columns = [col for col in result_df.columns if col.endswith('_std')]
    result_df = result_df.drop(columns=std_columns, errors='ignore')
    
    # Reorder columns to match the latex table structure
    base_columns = ['model', 'n_patches', 'backbone', 'input', 'weighting']
    
    if is_dense_capt:
        score_columns = ['map_score', 'METEOR', 'Bleu_4', 'ROUGE_L', 'CIDEr', 'SPICE', 'RefPAC-S', 'CLIP-S', 'PAC-S']
    else:
        score_columns = ['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE', 'RefPAC-S', 'CLIP-S', 'PAC-S']
    
    time_columns = ['avg_inference_time_per_image']
    
    # Build final column order, only including columns that exist
    final_columns = []
    for col_list in [base_columns, score_columns, time_columns]:
        final_columns.extend([col for col in col_list if col in result_df.columns])
    
    # Add any remaining columns that weren't in our predefined lists
    remaining_columns = [col for col in result_df.columns if col not in final_columns]
    final_columns.extend(remaining_columns)
    
    # Reorder the dataframe
    result_df = result_df[final_columns]
    
    return result_df


def print_latex_table(df : pd.DataFrame, num_digits : int = 1, is_dense_capt = False):
    
    latex_rows = []

    prev_model = None

    for _, row in df.iterrows():
        model, n_patches, backbone = row['model'], row['n_patches'], row['backbone'] #get_model_infos(row['model_name'])
        input_type = row['input']
        weighting = row['weighting']

        #handle num digits:
        def format_with_std(value, std_value, digits):
            try:
                if pd.isna(value) or value == '':
                    return ""
                val_float = float(value)
                if pd.isna(std_value) or std_value == '':
                    return f"{val_float:.{digits}f}"
                std_float = float(std_value)
                return f"{val_float:.{digits}f}±{std_float:.{digits}f}"
            except (ValueError, TypeError):
                return f"{value}" if not pd.isna(value) and value != '' else ""
        
        if num_digits == 0:
            if is_dense_capt:
                map = f"{row['map_score']:.2f}"
            bleu_4 = format_with_std(row['Bleu_4'], row.get('Bleu_4_std', ''), 0)
            meteor = format_with_std(row['METEOR'], row.get('METEOR_std', ''), 0)
            rouge = format_with_std(row['ROUGE_L'], row.get('ROUGE_L_std', ''), 0)
            cider = format_with_std(row['CIDEr'], row.get('CIDEr_std', ''), 0)
            spice = format_with_std(row['SPICE'], row.get('SPICE_std', ''), 0)
            refpac = format_with_std(row['RefPAC-S'], row.get('RefPAC-S_std', ''), 0)
            clips = format_with_std(row.get('CLIP-S', ''), row.get('CLIP-S_std', ''), 0)
            pacs = format_with_std(row.get('PAC-S', ''), row.get('PAC-S_std', ''), 0)
            inference_time = format_with_std(row.get('avg_inference_time_per_image', ''), row.get('std_inference_time_per_image', ''), 0)
        elif num_digits == 1:
            if is_dense_capt:
                map = f"{row['map_score']:.2f}"
            bleu_4 = format_with_std(row['Bleu_4'], row.get('Bleu_4_std', ''), 1)
            meteor = format_with_std(row['METEOR'], row.get('METEOR_std', ''), 1)
            rouge = format_with_std(row['ROUGE_L'], row.get('ROUGE_L_std', ''), 1)
            cider = format_with_std(row['CIDEr'], row.get('CIDEr_std', ''), 1)
            spice = format_with_std(row['SPICE'], row.get('SPICE_std', ''), 1)
            refpac = format_with_std(row['RefPAC-S'], row.get('RefPAC-S_std', ''), 1)
            clips = format_with_std(row.get('CLIP-S', ''), row.get('CLIP-S_std', ''), 1)
            pacs = format_with_std(row.get('PAC-S', ''), row.get('PAC-S_std', ''), 1)
            inference_time = format_with_std(row.get('avg_inference_time_per_image', ''), row.get('std_inference_time_per_image', ''), 3)
        elif num_digits == 2:
            if is_dense_capt:
                map = f"{row['map_score']:.2f}"
            bleu_4 = format_with_std(row['Bleu_4'], row.get('Bleu_4_std', ''), 2)
            meteor = format_with_std(row['METEOR'], row.get('METEOR_std', ''), 2)
            rouge = format_with_std(row['ROUGE_L'], row.get('ROUGE_L_std', ''), 2)
            cider = format_with_std(row['CIDEr'], row.get('CIDEr_std', ''), 2)
            spice = format_with_std(row['SPICE'], row.get('SPICE_std', ''), 2)
            refpac = format_with_std(row['RefPAC-S'], row.get('RefPAC-S_std', ''), 2)
            clips = format_with_std(row.get('CLIP-S', ''), row.get('CLIP-S_std', ''), 2)
            pacs = format_with_std(row.get('PAC-S', ''), row.get('PAC-S_std', ''), 2)
            inference_time = format_with_std(row.get('avg_inference_time_per_image', ''), row.get('std_inference_time_per_image', ''), 3)
        else:
            if is_dense_capt:
                map = f"{row['map_score']:.2f}"
            bleu_4 = format_with_std(row['Bleu_4'], row.get('Bleu_4_std', ''), 3)
            meteor = format_with_std(row['METEOR'], row.get('METEOR_std', ''), 3)
            rouge = format_with_std(row['ROUGE_L'], row.get('ROUGE_L_std', ''), 3)
            cider = format_with_std(row['CIDEr'], row.get('CIDEr_std', ''), 3)
            spice = format_with_std(row['SPICE'], row.get('SPICE_std', ''), 3)
            refpac = format_with_std(row['RefPAC-S'], row.get('RefPAC-S_std', ''), 3)
            clips = format_with_std(row.get('CLIP-S', ''), row.get('CLIP-S_std', ''), 3)
            pacs = format_with_std(row.get('PAC-S', ''), row.get('PAC-S_std', ''), 3)
            inference_time = format_with_std(row.get('avg_inference_time_per_image', ''), row.get('std_inference_time_per_image', ''), 3)

        if not is_dense_capt:
           latex_row = f"{model} & {n_patches} & {backbone} & {input_type} & {weighting} & {bleu_4} & {meteor} & {rouge} & {cider} & {spice} & {refpac} & {clips} & {pacs} & {inference_time} \\\\"
        else:
            latex_row =  f"{model} & {n_patches} & {backbone} & {input_type} & {weighting} & {map} & {meteor} & {bleu_4} & {rouge} & {cider} & {spice} & {refpac} & {clips} & {pacs} & {inference_time} \\\\"
        if prev_model is not None and prev_model != model:
            latex_rows.append(r"\midrule")
        prev_model = model
        latex_rows.append(latex_row)

    if not is_dense_capt:
        table_columns_info = "{lclcc*8r}"
    else:
        table_columns_info = "{lclcc*9r}"

    # Stampa l'intera tabella
    print(r"""\begin{table*}
    \centering
    \resizebox{\textwidth}{!}{
    \begin{tabular}"""+table_columns_info+"""
    \toprule""")
    if not is_dense_capt:
        print(r"""
        Model & \# Patches & Backbone & Input & Weighting & B & M & R & C & S & P & CLIP-S & PAC-S & Time (s) \\
    \midrule""")
    else:
        print(r"""
        Model & \# Patches & Backbone & Input & Weighting & mAP & M & B & R & C & S & P & CLIP-S & PAC-S & Time (s) \\
    \midrule""")
    for row in latex_rows:
        print(row)
    print(r"""\bottomrule
    \end{tabular}}
    \end{table*}""")
