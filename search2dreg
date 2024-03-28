# -*- coding: utf-8 -*-
#%%--------------------------------------------------

import os, sys
import numpy as np
import random
import time
import json
import matplotlib.pyplot as plt
import tqdm
import torch

#%%--------------------------------------------------

def generate_standard_elips(N_samples = 10, a= 1,b = 1):
    radius = 0.5
    center = 0
    N_samples1 = int(N_samples/2 - 1)
    N_samples2 = N_samples - N_samples1
    x1 = np.random.uniform((center-radius)*a,(center+radius)*a, size = N_samples1)
    x1_ordered = np.sort(x1)
    y1 = center + b*np.sqrt(radius**2 - ((x1_ordered-center)/a)**2)
    x2 =  np.random.uniform((center-radius)*a,(center+radius)*a, size = N_samples - N_samples1)
    x2_ordered = -np.sort(-x2) #the minus sign to sort descindingly
    y2 = center - b*np.sqrt(radius**2 - ((x2_ordered-center)/a)**2)
    x = np.concatenate([x1_ordered, x2_ordered], axis=0)
    y = np.concatenate([y1, y2], axis=0)
    return x, y

def destandarize_point(d, dim=128):
    return dim*(d + 0.5)

def To_pointcloud(x,y,z=0):
    N_points = x.shape[0]
    point_cloud = np.zeros([N_points,3])
    point_cloud[:,0] = x
    point_cloud[:,1] = y
    if not z==0:
        point_cloud[:,2] = z
    return point_cloud

def To_xyz(point_cloud):
    x = point_cloud[:,0]
    y = point_cloud[:,1]
    z = point_cloud[:,2]
    return x,y,z

x_t,y_t = generate_standard_elips(N_samples = 30, a= 0.8,b = 1.5)
pointcloud_target = To_pointcloud(x_t,y_t)

#%%--------------------------------------------------
#Generate_rigid_transformation

def random_rigid_transformation(dim=2):
    #dim = 4
    rotation_x = 0
    rotation_y = 0
    rotation_z = random.uniform(0, 2)*np.pi
    translation_x = random.uniform(-1, 1)*dim
    translation_y = random.uniform(-1, 1)*dim
    translation_z = 0
    reflection_x = random.sample([-1,1],1)[0]
    reflection_y = random.sample([-1,1],1)[0]
    reflection_z = 1
    Rotx = np.array([[1,0,0],
                     [0,np.cos(rotation_x),-np.sin(rotation_x)],
                     [0,np.sin(rotation_x),np.cos(rotation_x)]])
    Roty = np.array([[np.cos(rotation_y),0,np.sin(rotation_y)],
                     [0,1,0],
                     [-np.sin(rotation_y),0,np.cos(rotation_y)]])
    Rotz = np.array([[np.cos(rotation_z),-np.sin(rotation_z),0],
                     [np.sin(rotation_z),np.cos(rotation_z),0],
                     [0,0,1]])
    Rotation = np.matmul(Rotz, np.matmul(Roty,Rotx))
    Reflection = np.array([[reflection_x,0,0],[0,reflection_y,0],[0,0,reflection_z]])
    Translation = np.array([translation_x,translation_y,translation_z])
    RefRotation = np.matmul(Reflection,Rotation)
    return RefRotation, Translation

dim = 1.5
RefRotation, Translation = random_rigid_transformation(dim=dim)
pointcloud_source = np.matmul(RefRotation, pointcloud_target.T).T + Translation
x_s, y_s, z_s = To_xyz(pointcloud_source)

#%%--------------------------------------------------
#Satndardize
#Step1: center both
PC1_mean = np.mean(pointcloud_source, axis=0)
pointcloud_source_norm = pointcloud_source - PC1_mean

PC2_mean = np.mean(pointcloud_target, axis=0)
pointcloud_target_norm = pointcloud_target - PC2_mean

x_sn, y_sn, z_sn = To_xyz(pointcloud_source_norm)
x_tn, y_tn, z_tn = To_xyz(pointcloud_target_norm)

pointcloud_source_norm_torch = torch.tensor(pointcloud_source_norm, requires_grad=False).to(torch.float32)
pointcloud_target_norm_torch = torch.tensor(pointcloud_target_norm, requires_grad=False).to(torch.float32)


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x_s, y_s, color ='black',marker='x', linewidth = 1)
ax.plot(x_t, y_t, color ='red',marker='x', linewidth = 1)
#ax.plot(x_sn, y_sn, color ='black',marker='x', linewidth = 1)
#ax.plot(x_tn, y_tn, color ='red',marker='x', linewidth = 1)
ax.set_aspect('equal', adjustable ='box')
ax.set_xlim([-dim*1.25,dim*1.25])
ax.set_ylim([-dim*1.25,dim*1.25])


##--------------------------------------
##--------------------------------------
#Search
##--------------------------------------
##--------------------------------------
##--------------------------------------


def rigid_2Dtransformation(prdction):
    #prediction = [rotationz, reflectionx, reflectiony, translationx, translationy]
    N_examples = prdction['rotation'].shape[0]
    Translation = prdction['translation']
    Reflection = torch.zeros([N_examples,3,3])#
    Reflection[:,0,0] = prdction['reflection'][:,0]
    Reflection[:,1,1] = prdction['reflection'][:,1]
    Reflection[:,2,2] = 1.
    rotation_z = prdction['rotation'][:,2]
    Rotation = torch.zeros([N_examples,3,3])#np.repeat(np.eye(3)[None,:,:],N_examples, axis=0))
    Rotation[:,0,0] = torch.cos(rotation_z)
    Rotation[:,1,1] = torch.cos(rotation_z)
    Rotation[:,0,1] = -torch.sin(rotation_z)
    Rotation[:,2,2] = 1.
    Rotation[:,1,0] = torch.sin(rotation_z)
    RefRotation = torch.matmul(Reflection,Rotation)
    return RefRotation, Translation

def batch_hausdorff_prcnt_distance(batch_point_cloud1, point_cloud2, percentile = 0.95):
    assert point_cloud2.shape[0]==3
    assert batch_point_cloud1.shape[1]==3
    distances = torch.norm(batch_point_cloud1[:, :, None,:] - point_cloud2[None, :, :,None], dim=1)
    dists1 = torch.min(distances, dim=1).values
    dists2 = torch.min(distances, dim=2).values
    # Calculate the 95th percentile distance
    percentile_95 = torch.quantile(torch.cat([dists1, dists2],axis=1), percentile, interpolation='linear', dim=1)
    return percentile_95

def HDloss(prd, pointcloud_source_norm_torch,pointcloud_target_norm_torch, percentile = 0.95):
    A, b = rigid_2Dtransformation(prd)
    point_cloud_wrapped = torch.matmul(A, pointcloud_source_norm_torch.T) + b[:,:,None]
    loss = batch_hausdorff_prcnt_distance(point_cloud_wrapped, pointcloud_target_norm_torch.T, percentile)
    return loss

def wrap_pointcloud(record, pointcloud_source):
    #normalize first
    PC1_mean = np.mean(pointcloud_source, axis=0)
    pointcloud_source_norm = pointcloud_source - PC1_mean
    pointcloud_source_norm_torch = torch.tensor(pointcloud_source_norm, requires_grad=False).to(torch.float32)
    # find Tx
    A, b = rigid_2Dtransformation(record)
    point_cloud_wrapped = torch.matmul(A, pointcloud_source_norm_torch.T) + b[:,:,None]
    return point_cloud_wrapped

# Create a search dataset
def create_2d_permutations(precision, angle_start, angle_end, dist_start,dist_end):
    # Define three lists of numbers
    rotationX_list = np.array([0]) ##[1, 2, 3]
    rotationY_list = np.array([0]) ##[1, 2, 3]
    rotationZ_list = np.arange(angle_start, angle_end, precision*(angle_end-angle_start) )
    #-------------------
    translationX_list = np.arange(dist_start,dist_end,precision*(dist_end-dist_start))
    translationY_list = np.arange(dist_start,dist_end,precision*(dist_end-dist_start))
    translationZ_list = np.array([0]) ##[1, 2, 3]
    #-------------------
    reflectionX_list = np.array([1,-1])
    reflectionY_list = np.array([1,-1])
    reflectionZ_list = np.array([1])
    #-------------------
    # Create grids for each list
    g1,g2,g3,g4,g5,g6,g7,g8,g9 = np.meshgrid(rotationX_list, rotationY_list, rotationZ_list,
                                            translationX_list, translationY_list, translationZ_list,
                                            reflectionX_list,reflectionY_list,reflectionZ_list,
                                            indexing='ij')
    # Reshape the grids into 1D arrays and stack them horizontally
    permutations = np.column_stack((g1.ravel(), g2.ravel(),g3.ravel(),g4.ravel(),g5.ravel(),
                                    g6.ravel(),g7.ravel(),g8.ravel(),g9.ravel() ))
    print(permutations.shape)
    return permutations

def find_candidate(search_grid, N_batches, plot_solution = False,
                   loss_boundary = 10000000000000, N_candidates=3):
    candiates=[]
    Losses_hist = []
    Best_losses = []
    N_records = search_grid.shape[0]
    Batch_size = int(N_records//N_batches)
    for index in tqdm.tqdm(range(N_batches)):
        start = int(index*Batch_size)
        end = int((index+1)*Batch_size)
        permutations_torch = torch.tensor(search_grid).to(torch.float32)
        X_batch = {'rotation':permutations_torch[start:end,0:3],
                'translation':permutations_torch[start:end,3:6],
                'reflection':permutations_torch[start:end,6:9]}
        search_loss = HDloss(X_batch, pointcloud_source_norm_torch,pointcloud_target_norm_torch)
        Min_loss = torch.min(search_loss)
        Losses_hist.append(Min_loss.item()) 
        if Min_loss<loss_boundary:
            Best_loss = Min_loss
            index_min_loss = torch.argmin(search_loss)
            Best_record = {'rotation':X_batch['rotation'][index_min_loss:index_min_loss+1],
                    'translation':X_batch['translation'][index_min_loss:index_min_loss+1],
                    'reflection':X_batch['reflection'][index_min_loss:index_min_loss+1]}
            print('Best_loss:', Best_loss)
            if len(candiates)>=N_candidates:
                candiates.pop(0)
                Best_losses.pop(0)
            candiates.append(Best_record)
            Best_losses.append(Best_loss)
            if plot_solution:
                point_cloud_solution = wrap_pointcloud(Best_record, pointcloud_source)
                x_solution, y_solution, z_solution = To_xyz(point_cloud_solution)
                fig = plt.figure()
                ax = fig.add_subplot()
                #ax.plot(x_s,y_s, color ='black',marker='x',linestyle='-' , linewidth = 1)
                ax.plot(x_t, y_t, color ='red',marker='x',linestyle='--', linewidth = 1)
                ax.plot(x_solution[0], y_solution[0], color ='blue',marker='x',linestyle='-.', linewidth = 1)
                ax.set_aspect('equal', adjustable ='box')
                ax.set_xlim([-dim*1.25,dim*1.25])
                ax.set_ylim([-dim*1.25,dim*1.25])
                #plt.close()
    return candiates, Best_losses

distances = pointcloud_source_norm[:, None,:] - pointcloud_target_norm[None, :, :]
dists_max0 = np.max(distances)
#dists_max0 = batch_hausdorff_prcnt_distance(pointcloud_source_norm_torch.T[None, :, :], pointcloud_target_norm_torch.T, percentile=0.95)
dist_start = -dists_max0/2
dist_end = dists_max0/2
angle_start = -np.pi
angle_end = np.pi
dists_step = dist_end - dist_start
angle_step= angle_end - angle_start
precision0= 0.1
margin = 0.7
N_steps = 5
N_filtered_candidates=3

search_grid0 = create_2d_permutations(precision0, angle_start, angle_end, dist_start,dist_end)

N_batches = 4
N_records = search_grid0.shape[0]
print( 'N_records =', N_records, ', Batch_size = ', int(N_records//N_batches))
candiates, Best_losses = find_candidate(search_grid0, N_batches, plot_solution=True,
                   loss_boundary = 10000000000000, N_candidates=N_filtered_candidates)
loss_boundary = np.max(Best_losses)
bestLoss_ever = 1000000000
for step in range(N_steps):
    dists_step = dists_step*precision0*(1+margin)
    angle_step = angle_step*precision0*(1+margin)
    New_candidates = []
    New_best_losses = []
    for region in range(len(candiates)):
        dist_start = torch.max(candiates[region]['translation'])-dists_step
        dist_end = torch.max(candiates[region]['translation'])+dists_step
        angle_start = torch.max(candiates[region]['rotation'])-angle_step
        angle_end = torch.max(candiates[region]['rotation'])+angle_step
        search_grid = create_2d_permutations(precision0, angle_start, angle_end, dist_start,dist_end)
        candiates_i,Best_losses_i = find_candidate(search_grid, N_batches, plot_solution = False,
                       loss_boundary = loss_boundary, N_candidates=N_filtered_candidates)
        # select best 4 candiates out of 9 nominated candidates (3selectedx3regions) in the first step and
        # 4 candidates out of 12 (3selectedx4 regions) after
        #here we save 4 candidates to insure that the candiates from two regions at least
        # since max 3 candiates were selected from one region, hence the remaining candidate
        #at least will be from a different region
        for candidate_no in range(len(Best_losses_i)):
            if Best_losses_i[candidate_no]<= loss_boundary:
                New_candidates.append(candiates_i[candidate_no])
                New_best_losses.append(Best_losses_i[candidate_no])
                if Best_losses_i[candidate_no]<bestLoss_ever:
                    best_candidate_ever = candiates_i[candidate_no]
                    bestLoss_ever = Best_losses_i[candidate_no]
                if len(New_candidates)>N_filtered_candidates+1:
                    New_candidates.pop(0)
                    New_best_losses.pop(0)
                    loss_boundary = np.max(New_best_losses)
    candiates=New_candidates

print(best_candidate_ever)
print(bestLoss_ever)

#-------------------------------
#Evaluate
HDloss(best_candidate_ever, pointcloud_source_norm_torch, pointcloud_target_norm_torch, percentile = 0.95)
point_cloud_solution = wrap_pointcloud(best_candidate_ever, pointcloud_source)
x_solution, y_solution, z_solution = To_xyz(point_cloud_solution)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x_s,y_s, color ='black',marker='x',linestyle='-' , linewidth = 1)
ax.plot(x_t, y_t, color ='red',marker='x',linestyle='--', linewidth = 1)
ax.plot(x_solution[0], y_solution[0], color ='blue',marker='x',linestyle='-.', linewidth = 1)

ax.set_aspect('equal', adjustable ='box')
ax.set_xlim([-dim*1.25,dim*1.25])
ax.set_ylim([-dim*1.25,dim*1.25])


