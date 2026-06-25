import timm


legacy_vit_model_to_out_indices = {
    'vit_base_patch16_dinov3.lvd1689m': (7,),
    'vit_large_patch16_dinov3.lvd1689m': (17,),
    'vit_huge_plus_patch16_dinov3.lvd1689m': (23,),
    'vit_7b_patch16_dinov3.lvd1689m': (29,),
}


def _create_dino_v3(model_name, out_indices=None):
    if out_indices is None:
        out_indices = legacy_vit_model_to_out_indices[model_name]
    return timm.create_model(
        model_name=model_name,
        features_only=True,
        dynamic_img_size=True,
        out_indices=out_indices,
    )


def dino_v3_b(out_indices=None):
    return _create_dino_v3('vit_base_patch16_dinov3.lvd1689m', out_indices=out_indices)

def dino_v3_l(out_indices=None):
    return _create_dino_v3('vit_large_patch16_dinov3.lvd1689m', out_indices=out_indices)

def dino_v3_h_plus(out_indices=None):
    return _create_dino_v3('vit_huge_plus_patch16_dinov3.lvd1689m', out_indices=out_indices)

def dino_v3_7b(out_indices=None):
    return _create_dino_v3('vit_7b_patch16_dinov3.lvd1689m', out_indices=out_indices)
