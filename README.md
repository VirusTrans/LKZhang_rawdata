# LKZhang_rawdata

This repository contains the code that accompanies the paper introducing "*TriLock-FISH: Spatial discrimination of influenza A virus vRNA, cRNA, and mRNA in single cells via a split-probe ligation strategy*".

## Code

### Python environment

We recommend running the code in VS Code. A GPU may be required. 

To re-create the python environments with `conda` run:

```bash
conda env create -f envs/TriLock-FISH
```

Load the following packages.

```bash
import os
import numpy as np
import pandas as pd
from skimage import io, exposure
from cellpose import models
import matplotlib.pyplot as plt
import skimage.io
from skimage.color import label2rgb
import seaborn as sns
from skimage.measure import label, regionprops
from sklearn.neighbors import KDTree
from ufish.api import UFish

ufish = UFish()
ufish.load_weights()
```

### TIF format Picture Analyses
First step
```bash
def cc_centroids(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == bool:
        mask = label(mask)
    ccs = regionprops(mask)
    centroids, labels = [], []
    for cc in ccs:
        centroids.append(cc.centroid)
        labels.append(cc.label)
    return np.array(centroids), np.array(labels)

def assign_spots(
        spots: np.ndarray,
        mask: np.ndarray,
        dist_th: float,
        max_iter: int = 0,
        iter_dist_step: float = 10,
        ) -> np.ndarray:
    assert len(mask.shape) in (2, 3)
    centers, labels = cc_centroids(mask)
    assert centers.shape[1] == len(mask.shape)
    search_map = np.zeros(  # label(index) -> center(value)
        (labels.max(), centers.shape[1]), dtype=centers.dtype)
    search_map[labels - 1] = centers
    pos_each_axes = np.where(mask > 0)
    pos_ = np.c_[pos_each_axes]
    tree = KDTree(pos_)
    dist, idx = tree.query(spots)
    dist, idx = np.concatenate(dist), np.concatenate(idx)
    clost = pos_[idx, :]
    if centers.shape[1] == 2:
        mask_val = mask[clost[:, 0], clost[:, 1]]
    else:
        mask_val = mask[clost[:, 0], clost[:, 1], clost[:, 2]]
    res = search_map[mask_val - 1]
    res[dist > dist_th, :] = np.nan

    # Iteratively find neighbors
    for _ in range(max_iter):
        nan_idxs = np.where(np.isnan(res))[0]
        if nan_idxs.shape[0] == 0:
            break
        non_nan_idxs = np.where(~np.isnan(res))[0]
        non_nan_res = res[non_nan_idxs]
        non_nan_pos = spots[non_nan_idxs, :]
        tree = KDTree(non_nan_pos)
        nan_pos = spots[nan_idxs, :]
        dist, idx = tree.query(nan_pos)
        dist, idx = np.concatenate(dist), np.concatenate(idx)
        nan_res = non_nan_res[idx]
        nan_res[dist > iter_dist_step] = np.nan
        res[nan_idxs] = nan_res

    return res
```

Second step

```bash
def gene_to_cell(
        gene_table,
        cell_mask,
        dist_thresh= 30.0,
        max_iter= 5,
        dist_step= 10.0,
        ) :  # type: ignore

    dims = [c for c in gene_table.columns if c.startswith('dim')]
    spots = gene_table[dims].values
    related_centers = assign_spots(
        spots, cell_mask, dist_thresh,
        max_iter=max_iter, iter_dist_step=dist_step)
    df_center = pd.DataFrame({
        d: related_centers[:, i]
        for i, d in enumerate(dims)
    }).applymap("{:.2f}".format)
    center_ser = "(" + df_center.iloc[:, 0]
    for i in range(1, len(dims)):
        center_ser += (", " + df_center.iloc[:, i])
    center_ser += ")"
    df = pd.DataFrame({
        'gene': gene_table['gene'].values,
        'cell': center_ser.values
    })
    counts = df.value_counts()
    exp_mat = counts.unstack(fill_value=0).T
    return exp_mat, related_centers
```

Third step

```bash
from skimage.filters import gaussian

def processed_img(file_path, dist_thresh,i):
    print(file_path)
    #get_digital_data
    img = io.imread(file_path)
    prespots, _ = ufish.predict(img[:,:, 1:4], axes='yxc', intensity_threshold=i) # For 4-channel TIF only
    model = models.Cellpose(gpu=True, model_type='nuclei')

    dapi = gaussian(img[:,:,0], sigma=2)  # Adjust the intensity of the added Gaussian noise based on the image background; high-quality input images are recommended. 
  
    masks, flows, styles, diams = model.eval(dapi, 
                                             channels=[[0,0]], 
                                             diameter=100,
                                             flow_threshold=4.0,
                                             cellprob_threshold=0.0,
                                             niter=500)
    #plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img.mean(axis=2), cmap='gray')
    axes[0].axis('off')    
    axes[1].imshow(img[:,:,0], cmap='gray')
    axes[1].imshow(masks, cmap='tab20', alpha=0.5)
    axes[1].axis('off')

    plt.show()

    gene_mapping = {0: 'vrna', 1: 'crna', 2: 'mrna'}
    prespots['gene'] = prespots['axis-2'].map(gene_mapping)
    table=pd.DataFrame()
    table["dim_0"] = prespots["axis-0"]
    table["dim_1"] = prespots["axis-1"]
    table['gene'] = prespots['gene']
    
    #delet_0_spots_label
    table['mask_label'] = masks[table['dim_0'].astype(int), table['dim_1'].astype(int)]
    mask_counts = table.groupby('mask_label').size()
    # print("mask_counts:\n", mask_counts)
    
    labels_to_keep = mask_counts[mask_counts >2].index
    # print("labels_to_keep:\n", labels_to_keep)

    masks[~np.isin(masks, labels_to_keep)] = 0
    if np.count_nonzero(masks) == 0:
        print(f"跳过文件: {file_path}")
        spot_csv_path = 0

    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(img[:,:,0], cmap='gray')
        axes[0].axis('off')    
        axes[1].imshow(img[:,:,0], cmap='gray')
        axes[1].imshow(masks, cmap='tab20', alpha=0.5)
        axes[1].axis('off')

        plt.show()
        
        gene2cell,related_ct = gene_to_cell(table,masks,dist_thresh=dist_thresh)
        table["y"] = related_ct[:,0]
        table["x"] = related_ct[:,1]
        table.dropna(subset=['y'], inplace=True)

        dir = file_path.split('/')[1]
        filename = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = f'{dir}_results'
        os.makedirs(output_dir, exist_ok=True)
        spot_csv_path = f"./{output_dir}/spots_{filename}.csv"
        table.to_csv(spot_csv_path,index=False)

        matrix_output_path = f'./{output_dir}/check_matrix_{filename}.csv'
        gene2cell.to_csv(matrix_output_path)

        io.imsave(f'./{output_dir}/mask.tif', masks)
        
    return spot_csv_path, masks
```


Fourth step

```bash
def count_genes_per_coordinate(csv_path, output_path=None):
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取 {csv_path}")
    except FileNotFoundError:
        print(f"文件 {csv_path} 未找到。请检查文件路径。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    required_columns = {'y', 'x', 'gene', 'mask_label'}
    if not required_columns.issubset(df.columns):
        print(f"输入的 CSV 文件缺少必要的列: {required_columns - set(df.columns)}")
        return

    df['y'] = df['y'].round(2)
    df['x'] = df['x'].round(2)
    df['coordinate'] = "(" + df['y'].astype(str) + "," + df['x'].astype(str) + ")"

    gene_counts = df.groupby(['coordinate', 'gene']).size().reset_index(name='count')
    result_df = gene_counts.pivot(index='coordinate', columns='gene', values='count').fillna(0).astype(int)

    df['location'] = np.where(df['mask_label'] == 0, 'cyto', 'nuclei')

    df_loc_counts = (
        df.groupby(['coordinate', 'gene', 'location'])
          .size()
          .unstack(['gene', 'location'], fill_value=0)
    )

    df_loc_counts.columns = [f"{gene.lower()}_{loc}" for gene, loc in df_loc_counts.columns]

    wanted_cols = [
        'crna_nuclei', 'mrna_nuclei', 'vrna_nuclei',
        'crna_cyto',   'mrna_cyto',   'vrna_cyto'
    ]
    for col in wanted_cols:
        if col not in df_loc_counts.columns:
            df_loc_counts[col] = 0

    df_loc_counts = df_loc_counts[wanted_cols]
    result_df = result_df.join(df_loc_counts, how='left').fillna(0).astype(int)
    if output_path == 'True':
        try:
            dir = csv_path.split('/')[1]
            filename = os.path.splitext(os.path.basename(csv_path))[0]
            tmp_name = filename.split('ots_')[1]
            output_name = f'matrix_{tmp_name}.csv'
            output_path = f'./{dir}/{output_name}'
            result_df.to_csv(output_path)
            print(f"统计结果已保存到 {output_path}")
        except Exception as e:
            print(f"保存文件时发生错误: {e}")

    return result_df

```


Fifth step 

```bash
input_dir = 'data/TriLock_FISH'
files = os.listdir(input_dir)

df_list= []

for file in files:
    input = os.path.join(input_dir, file)
    spots_csv_path, masks = processed_img(input, 120, 0.25)
    if spots_csv_path != 0:
        result = count_genes_per_coordinate(spots_csv_path, output_path='True')
        print(result)
        result['file_name'] = file
        df_list.append(result)
    else:
        continue
final_df = pd.concat(df_list, axis=0)
final_df.to_csv('TriLock_FISH.csv')

```
Key Point:
- Change the storage path of the original TIF images in `input_dir = 'data/TriLock_FISH`. The TIF image data should be placed in the TriLock_FISH folder under data; the folder name can be customized. 

- Output file is ` TriLock_FISH.csv `.

  
### Visualization for quality control 

```bash

def visualize_gene_expression(csv_path, mask_path, base_image_path, output_image_path=None):
    
    df = pd.read_csv(csv_path)
    print(f"成功读取CSV文件: {csv_path}")

    mask = skimage.io.imread(mask_path)
    print(f"成功读取mask文件: {mask_path}")

    img = skimage.io.imread(base_image_path)
    img = img[:,:,0]
    print(f"成功读取基底图像文件: {base_image_path}")
    

    required_columns = {'dim_0', 'dim_1', 'gene', 'mask_label', 'y', 'x'}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        print(f"CSV文件缺少必要的列: {missing_cols}")
        return

    df['y'] = df['y'].round(2)
    df['x'] = df['x'].round(2)
    df['coordinate'] = "(" + df['y'].astype(str) + "," + df['x'].astype(str) + ")"

    mask_centers = df[['mask_label', 'y', 'x']].drop_duplicates()
    print(f"找到 {len(mask_centers)} 个唯一的mask中心坐标。")

    fig, ax = plt.subplots(figsize=(12, 12))

    if img.ndim == 3:
        img_gray = img[:, :, 0]
    else:
        img_gray = img
    ax.imshow(img_gray, cmap='Blues')
    print("已绘制最底层的灰度图像。")
    
    # mask_rgb = label2rgb(mask, alpha=0.3, image=img_gray)
    # ax.imshow(mask_rgb)
    print("已绘制mask作为背景。")
    
    spot_x = df['dim_1'].values
    spot_y = df['dim_0'].values
    genes = df['gene'].values
    mask_labels = df['mask_label'].values

    sns.set_palette("colorblind")
    unique_genes = df['gene'].unique()
    num_genes = len(unique_genes)
    palette = sns.color_palette("colorblind", num_genes)
    gene_colors = {gene: palette[i % len(palette)] for i, gene in enumerate(unique_genes)}
    spot_colors = [gene_colors[gene] for gene in genes]
    
    scatter = ax.scatter(spot_x, spot_y, c=spot_colors, s=25, edgecolors='k', label='Gene Spots')
    print("已绘制基因表达spots。")

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=gene,
                          markerfacecolor=gene_colors[gene], markersize=10, markeredgecolor='k') 
               for gene in unique_genes]

    mask_center_handle = plt.Line2D([0], [0], marker='x', color='blue', linestyle='None',
                                    markersize=10, label='Mask Centers')

    for idx, row in df.iterrows():
        spot_x_coord = row['dim_1']
        spot_y_coord = row['dim_0']
        center_x = row['x']
        center_y = row['y']
        
        ax.plot([spot_x_coord, center_x], [spot_y_coord, center_y], 
                color='red', linewidth=0.5, alpha=0.35)
    
    print("已绘制连接线。")
    
    ax.scatter(mask_centers['x'], mask_centers['y'], c='blue', marker='x', s=50, label='Mask Centers')
    print("已绘制mask中心点。")
    
    ax.legend(handles=handles + [mask_center_handle], title='Genes and Mask Centers', loc='upper right')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Gene Assignment Visualization')
    
    if output_image_path:
        plt.savefig(output_image_path, bbox_inches='tight')
        print(f"可视化结果已保存到: {output_image_path}")
    
    plt.show()

if __name__ == "__main__":

    dir = 'TriLock_FISH'
    for file in os.listdir(dir):
        sample = file.split('.')[0]
        csv_path = f"./{dir}/{sample}.tif_results/spots_{sample}.csv"                 
        mask_path = f"./{dir}/{sample}.tif_results/mask.tif"     
        base_image_path = f"./data/{dir}/{sample}.tif"
        output_image_path = f"./assign/{sample}.pdf"   
        os.makedirs('assign', exist_ok=True)
        visualize_gene_expression(csv_path, mask_path, base_image_path, output_image_path)

```










