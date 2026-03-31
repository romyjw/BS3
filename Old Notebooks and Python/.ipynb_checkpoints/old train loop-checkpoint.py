'''

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


import implicit_reps
import importlib
importlib.reload(implicit_reps)
from implicit_reps import *




##################### define training loop ###########################

def train_batch_loop(
    lr=0.01,
    batch_size=10000,
    min_loss=0.0005,
    normals_reg_coeff=0.0,
    distortion_reg_coeff=0.0,
    batch_print_rate=100,
    batch_plot_rate=10,
    epoch_print_rate=1,
    epoch_save_rate=1,
    epoch_plot_rate=1,
    plot=False,
    flag='poly',
    just_onering=False,
    degree=3,
    max_epochs = 1000
):
    global optimizer, losses, pos_losses, normals_regs, times, epoch, loss, num_patches, eps, shape_id

    #for g in optimizer.param_groups:
    #   g['lr'] = lr

    if degree > bps.degree:
        raise ValueError("Sorry, you can't train this BPS with degree", degree, "because the BPS was only initialised with enough parameters for degree", bps.degree, "polynomials.")

    if just_onering==True:

        # If you want indices from a onering
        select_patch_indices = torch.tensor(
            bps.onerings[0]['triangles'], dtype=torch.long, device=bps.device
        )
    else:
        # If you want all triangle indices
        select_patch_indices = torch.arange(
            bps.F.shape[0], dtype=torch.long, device=bps.device
        )

    seqs = bps.F.shape[0]
    total_num_points = precomputed_training_data['equilateral_triangle_samples'].shape[1]

    

    #updated_target_train = target_train
    #print(updated_target_train.shape)

    while loss > min_loss and epoch<max_epochs:
        perm = torch.randperm(total_num_points)
        epoch_loss = 0.0

        for i in range(0, total_num_points, batch_size):
            batch_start = time.time()

            idx = perm[i:i + batch_size]
            

            optimizer.zero_grad()


            #output_train, onering_x = bps(batch_t, select_patch_indices=select_patch_indices, return_onering_coords=True, test_flag=flag)
            #check_for_nans(output_train, "output_train")

            precomputed_batch_data = bps.get_batch(precomputed_training_data, idx)

            batch_t = precomputed_batch_data['equilateral_triangle_samples']
            batch_t.requires_grad = True

            output_train = bps(precomputed_batch_data, degree=degree)


            selected_pts = output_train[select_patch_indices, :, :].to(device)


            #print("output_train device:", output_train.device)
            #print("select_patch_indices device:", getattr(select_patch_indices, "device", "not a tensor"))
            #print("selected_pts device:", selected_pts.device)

            cur_sdf = sdf(selected_pts.reshape(1, -1, 3), config_dict['shape_id'], squared=False, model=DEEPSDF_MODEL)
            cur_udf = abs(cur_sdf)

            position_loss = cur_udf.mean()

            


            if normals_reg_coeff > 0.0 or distortion_reg_coeff > 0.0:
                output_normals, jacobian = diffmod.compute_normals(out=output_train, wrt=batch_t, return_grad=True)
                #output_normals, jacobian = diffmod.compute_normals(out=output_train, wrt=onering_x, return_grad=True)
                
                check_for_nans(jacobian, "jacobian")
                check_for_nans(output_normals, "output_normals")


            if normals_reg_coeff > 0.0:
                

                grad_sdf = diffmod.gradient(out=cur_sdf.unsqueeze(-1), wrt=output_train).squeeze()
                check_for_nans(grad_sdf, "grad_sdf")

                normals_alignment = (output_normals * grad_sdf).sum(-1).mean()
                check_for_nans(normals_alignment, "normals_alignment")

                normals_reg = normals_reg_coeff * (-normals_alignment + 1.0)
                check_for_nans(normals_reg, "normals_reg")
            else:
                normals_reg = torch.tensor(0.0)


            if distortion_reg_coeff > 0.0:
                I_E, I_F, I_G = diffmod.compute_FFF(out=output_train, wrt=batch_t, jacobian=jacobian, normals=output_normals)
                trace = I_E + I_G
                det = I_E * I_G - I_F * I_F

                #distortion_reg = distortion_reg_coeff * ( trace / det.sqrt() ).mean()
                distortion_reg = distortion_reg_coeff * ( trace + trace / (det+eps) ).mean()

                
            else:
                distortion_reg = torch.tensor(0.0)    

            batch_loss = position_loss + normals_reg + distortion_reg
            check_for_nans(batch_loss, "batch_loss")


            scheduler.step()

            batch_loss.backward()

            # Optional: clip gradients
            #torch.nn.utils.clip_grad_norm_(bps.parameters(), max_norm=1.0)

            # Optional: check gradients for NaNs
            for name, param in bps.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"⚠️ NaNs in gradient of parameter: {name}")
                    raise ValueError(f"NaN gradient detected in {name}")

            optimizer.step()

            epoch_loss += batch_loss * batch_t.shape[1]

            batch_time = time.time() - batch_start

            times.append(batch_time)
            losses.append(batch_loss.detach().item())
            pos_losses.append(position_loss.detach().item())
            normals_regs.append(normals_reg.detach().item())
            distortion_regs.append(distortion_reg.detach().item())
            fractional_epochs.append(epoch + i / total_num_points)

            if (i//batch_size) % batch_plot_rate == 0 and plot:
                # the batch_plot_rate is the number of batches between each draw of the loss plot
                try:
                    plt.close(fig)
                except:
                    pass
                    
                clear_output(wait=True)

                # Plot losses
                fig, axs = plt.subplots(2, 1, figsize=(10, 6))

                axs[0].plot(fractional_epochs, np.log10(losses), label='loss')
                axs[0].plot(fractional_epochs, np.log10(pos_losses), label='pos loss')
                if normals_reg_coeff > 0:
                    axs[0].plot(fractional_epochs, np.log10(normals_regs), label='normals reg')
                if distortion_reg_coeff > 0:
                    axs[0].plot(fractional_epochs, np.log10(distortion_regs), label='distortion reg')
                    
                axs[0].legend()
                axs[0].set_xlabel('Epoch')
                axs[0].set_ylabel('Log10 of Loss')
                axs[0].set_title('Log10 of Losses')

                # write current losses to the plot in text

                axs[0].text(
                    0.8, 0.9, f"Batch Loss: {batch_loss.item():.8f}",
                    transform=axs[0].transAxes, ha='right', va='bottom',
                    fontsize=10, color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
                )
                axs[0].text(
                    0.8, 0.8, f"Batch Pos. Loss: {position_loss.item():.8f}",
                    transform=axs[0].transAxes, ha='right', va='bottom',
                    fontsize=10, color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
                )

                if normals_reg_coeff > 0:
                    axs[0].text(
                        0.8, 0.7, f"Batch Normals Reg.: {normals_reg.item():.8f}",
                        transform=axs[0].transAxes, ha='right', va='bottom',
                        fontsize=10, color='black',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
                    )

                if distortion_reg_coeff > 0:
                    axs[0].text(
                        0.8, 0.6, f"Batch Distortion Reg.: {distortion_reg.item():.8f}",
                        transform=axs[0].transAxes, ha='right', va='bottom',
                        fontsize=10, color='black',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
                    )

                axs[1].plot(fractional_epochs, times)
                axs[1].set_xlabel('Epoch')
                axs[1].set_title('Time per Batch')

                axs[1].text(
                    0.99, 0.90, f"Batch Time: {batch_time:.4f}s",
                    transform=axs[1].transAxes, ha='right', va='bottom',
                    fontsize=10, color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
                )

                plt.tight_layout()
                display(fig)
                #plt.close(fig)

                #Print out some extra info below the plots

                if normals_reg_coeff > 0:
                    print(f"Batch {i}, Loss: {batch_loss:.8f}, Pos. Loss: {position_loss:.8f}, "
                          f"Normals Reg: {(normals_reg_coeff * normals_reg):.8f}, Batch Time: {batch_time:.4f}s")
                else:
                    print(f"Batch {i}, Loss: {batch_loss:.8f}, Pos. Loss: {position_loss:.8f}, Batch Time: {batch_time:.4f}s")
    
                print(f"mean UDF: {cur_udf.mean():.8f}")
    
                total_batch_time = time.time() - batch_start
                print(f"Time inc plotting: {total_batch_time:.4f}s")
        
                primary_lr = optimizer.param_groups[0]['lr']
                print(f"Current Learning Rate: {primary_lr:.8f}")
    
    

        loss = torch.tensor(epoch_loss / total_num_points)

        if epoch % epoch_print_rate == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.8f}")

        if epoch % epoch_save_rate == 0:
            #bns.save_mlps('models/temp.pth')
            bps.save_poly_coeffs('models/temp.pth')
            print('Saved checkpoint.')

        epoch += 1
'''
        
