import learn2learn as l2l
import torch
from torch_geometric import seed_everything
from utilsfiletraining import *
from architecturenn import GCN, AtomMessagePassing
import os

def main_maml_training_loop(maml, train_loader, meta_valid_loader, scalers, 
                            optimizer_maml, cond_names, criterion, input_channels, flag_scale_target, flag_two_transforms, 
                            epochs=100, adaptation_steps=5,
                            device='cuda', savedir='.',
                            meta_batch_size=8, flag_clean_start=True,
                           ):
    train_losses = []
    valid_losses = []
    r2s_valid = []
    best_valid_loss = float('inf')
    if len(scalers) == 1: 
        scaler = scalers[0]
    if len(scalers) == 2: 
        scaler = scalers[0]
        scaler2 = scalers[1]

    if os.path.isfile(f'{savedir}/best_model_maml.pth') and not flag_clean_start:
        maml.module.load_state_dict(torch.load(f'{savedir}/best_model_maml.pth', weights_only=True))

    global current_epoch
    current_epoch = 0
    for epoch in tqdm(range(epochs)):   
        current_epoch = epoch
        task_batch = []

        maml.train()

        train_loss = 0
        r2 = 0
        
        for data in train_loader:
            task_batch.append(data)
            
            if len(task_batch) == meta_batch_size:
                break
        
        meta_loss = 0.0
        for data in task_batch:
            supportL, supportS, queryL, queryS = data
            
            support = (supportL.to(device), supportS.to(device))
            query = (queryL.to(device), queryS.to(device))

            model = maml.clone()
    
            for _ in range(adaptation_steps):
                if cond_names:
                    out, _ = model(support[0].x, support[0].edge_index, support[0].edge_attr, support[1].x, support[1].edge_index, support[1].edge_attr, support[0].batch, support[1].batch, support[0].temps)
                else:
                    out, _ = model(support[0].x, support[0].edge_index, support[0].edge_attr, support[1].x, support[1].edge_index, support[1].edge_attr, support[0].batch, support[1].batch)
                loss_s = criterion(out.squeeze(), support[0].y)
                model.adapt(loss_s)

            if cond_names:
                out, _ = model(query[0].x, query[0].edge_index, query[0].edge_attr, query[1].x, query[1].edge_index, query[1].edge_attr, query[0].batch, query[1].batch, query[0].temps)
            else:
                out, _ = model(query[0].x, query[0].edge_index, query[0].edge_attr, query[1].x, query[1].edge_index, query[1].edge_attr, query[0].batch, query[1].batch)
            loss_q = criterion(out.squeeze(), query[0].y)
            meta_loss += loss_q
    
        optimizer_maml.zero_grad()
        meta_loss.backward()
        optimizer_maml.step()
        task_batch = []

        train_loss += meta_loss.detach().cpu().item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
    
        with torch.no_grad():
            model_val = maml.clone()
            state_dict = model_val.module.state_dict()
            model_to_valid = GCN(
                AtomMessagePassing,
                input_channels = input_channels, input_channels2 = int(input_channels / 2),
                embedding_size = 128, hidden_dim = 256,
                linear_size = [256,128], add_params_num = len(cond_names))
            
            model_to_valid.load_state_dict(state_dict)
            model_to_valid = model_to_valid.to(device)
            
            model_to_valid.eval()
            valid_loss = 0
            y_true = []
            y_pred = []
            for data in meta_valid_loader:
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
                if cond_names:
                    out2, _ = model_to_valid(data[0].x, data[0].edge_index, data[0].edge_attr, data[1].x, data[1].edge_index, data[1].edge_attr, data[0].batch, data[1].batch, data[0].temps)
                else:
                    out2, _ = model_to_valid(data[0].x, data[0].edge_index, data[0].edge_attr, data[1].x, data[1].edge_index, data[1].edge_attr, data[0].batch, data[1].batch)
                loss_val = criterion(out2.squeeze(), data[0].y)
                valid_loss += loss_val.detach().cpu().item()
                y_true.extend(data[0].y.cpu().numpy())
                y_pred.extend(out2.squeeze().detach().cpu().numpy())
            valid_loss /= len(meta_valid_loader)
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
    
            if flag_scale_target:
                if flag_two_transforms:
                  y_true_unscaled = scaler2.inverse_transform(y_true.reshape(-1, 1)).flatten()
                  y_pred_unscaled = scaler2.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                  y_true_unscaled = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
                  y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                else:
                  y_true_unscaled = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
                  y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
            metric = RegressionMetric(y_true_unscaled, y_pred_unscaled)
            rmse = metric.get_metric_by_name('RMSE')['RMSE']
            r2 = metric.get_metric_by_name('R2')['R2']
            mae = metric.get_metric_by_name('MAE')['MAE']
            mare = float(np.mean(np.abs(y_true_unscaled - y_pred_unscaled) / np.abs(y_true_unscaled + 1e-8), axis=0))
        
            valid_losses.append(valid_loss)
            r2s_valid.append(r2)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.module.state_dict(), f'{savedir}/best_model_maml.pth')
                best_train_loss = train_loss
                best_valid_loss = valid_loss
                best_r2 = r2
                best_rmse = rmse
                best_mae = mae
                best_mare = mare
    return maml, train_losses, valid_losses, r2s_valid

def fine_tuning_experiment_run(df, 
                               random_state_adjust, 
                               maml, solute_name_dict, use_cross_val, use_adjust_val_subset, fold_to_use, 
                               compounds_names_to_test, similarity_vals,
                               input_channels, cond_names, savedir, device, scalers, flag_scale_target, flag_two_transforms,
                               validation_schema='random', 
                               adapt_selection_method='centroids',
                               epochs = 25,
                               flag_verbose=False, shot_sizes=[8, 16, 32, 64, 128],
                               flag_no_ft=False, 
                               test_all_sets=False,train_loader=None,valid_loader=None
                              ):
    
    num_metrics = 4

    _, _, full_test_loader = get_loaders(df, 'pre-training')

    model_to_adapt = GCN(
        AtomMessagePassing,
        input_channels = input_channels, input_channels2 = int(input_channels / 2),
        embedding_size = 128, hidden_dim = 256,
        linear_size = [256,128], add_params_num = len(cond_names))

    model = maml.clone()
    model_to_adapt.load_state_dict(torch.load(f'{savedir}/best_model_maml.pth', weights_only=False))
    model_to_adapt = model_to_adapt.to(device)

    if test_all_sets:
        print("Training Test : R2, RMSE, MAE, (MARE)")
        train_loader_nonbatched = PYGDataLoader(df[df['split'] == 0]['graphs'].values, batch_size=32768, shuffle=True, collate_fn=collate_fn)
        _ = test_model(model_to_adapt, train_loader_nonbatched, "Full Test", scalers, cond_names, flag_scale_target, flag_two_transforms, device)
        print("Validation Test : R2, RMSE, MAE, (MARE)")
        _ = test_model(model_to_adapt, valid_loader, "Full Test", scalers, cond_names, flag_scale_target, flag_two_transforms, device)

    print("Full Test : R2, RMSE, MAE, (MARE)")
    mse_test_full, _, _, rmse_test_full, r2_test_full, mae_test_full, mare_test_full = test_model(model_to_adapt, full_test_loader, "Full Test", scalers, cond_names, flag_scale_target, flag_two_transforms, device)

    if not compounds_names_to_test or flag_no_ft: return None
    
    res_tasks_np = []
    for comp_name in compounds_names_to_test:
        shot_sizes_expres = np.zeros((len(shot_sizes), num_metrics), dtype=float)
        for shot_size_i, shot_size in enumerate(shot_sizes):
            df = prepare_splits_fine_tuning(df, comp_name, validation_schema, 
                                            solute_name_dict, use_cross_val, use_adjust_val_subset, shot_size, fold_to_use,
                                            adapt_selection_method=adapt_selection_method, 
                                            random_state_adjust=random_state_adjust, n_clusters=4)
            df, tasks, tasks_dict, y = extract_tasks_and_targets(df)
            df = add_graphs_to_df(df, tasks)
            adapt_loader, valid_loader, test_loader, full_test_loader = get_loaders(df, 'fine-tuning')
            
            model = maml.clone()
            model_to_adapt = GCN(
                AtomMessagePassing,
                input_channels = input_channels, input_channels2 = int(input_channels / 2),
                embedding_size = 128, hidden_dim = 256,
                linear_size = [256,128], add_params_num = len(cond_names))
            
            model_to_adapt.load_state_dict(torch.load(f'{savedir}/best_model_maml.pth', weights_only=False))
            model_to_adapt = model_to_adapt.to(device)
    
            print(f"Task {comp_name} Test Zero Shot : R2, RMSE, MAE, (MARE)")
            mse_test_zero, _, _, rmse_test_zero, r2_test_zero, mae_test_zero, mare_test_zero = test_model(model_to_adapt, test_loader, "Zero-shot Test", scalers, cond_names, flag_scale_target, flag_two_transforms, device)
        
            model_to_adapt, train_losses_ft, valid_losses_ft, r2s_valid_ft = fine_tuning_loop(model_to_adapt, scalers, 
                                                                                              adapt_loader, valid_loader, cond_names, flag_scale_target, flag_two_transforms,
                                                                                              epochs = epochs, device=device, savedir=savedir)
            if flag_verbose: 
                plot_finetune_losses_vs_epoch(train_losses_ft, valid_losses_ft, r2s_valid_ft, savedir, flag_verbose)
            
            model_to_adapt.load_state_dict(torch.load(f'{savedir}/best_model.pth', weights_only=False))
            
            print(f"{comp_name} Adaptation - Validation - Test - Full Test: R2, RMSE, MAE, (MARE)")
            mse_train, y_true_adapt_par, y_pred_adapt_par, rmse_train, r2_train, mae_train, mare_train = test_model(model_to_adapt, adapt_loader, "Training", scalers, cond_names, flag_scale_target, flag_two_transforms, device)
            mse_valid, y_true_valid_par, y_pred_valid_par, rmse_valid, r2_valid, mae_valid, mare_valid = test_model(model_to_adapt, valid_loader, "Validation", scalers, cond_names, flag_scale_target, flag_two_transforms, device)
            mse_test, y_true_test_par, y_pred_test_par, rmse_test, r2_test, mae_test, mare_test = test_model(model_to_adapt, test_loader, "Test", scalers, cond_names, flag_scale_target, flag_two_transforms, device)
            mse_test_full_aft, _, _, rmse_test_full_aft, r2_test_full_aft, mae_test_full_aft, mare_test_full_aft = test_model(model_to_adapt, full_test_loader, "Full Test", scalers, cond_names, flag_scale_target, flag_two_transforms, device)
            
            if flag_verbose >= 2:
                plt.figure(figsize=(8, 5))
                sns.violinplot(data=[mse_train, mse_valid, mse_test], inner="point", palette=["blue", "orange", "green"])
                plt.xticks([0, 1, 2], ["Training", "Validation", "Test"])
                plt.ylabel("Mean Squared Error (MSE)")
                plt.title(f"{comp_name} Distribution of MSE Across Datasets")
                plt.show()

            if flag_verbose:
                plot_parity(y_true_adapt_par, y_pred_adapt_par, flag_verbose, savedir, f"Task Adapt set {comp_name} parity plot {shot_size} shot {random_state_adjust}")
                plot_parity(y_true_valid_par, y_pred_valid_par, flag_verbose, savedir, f"Task Validation set {comp_name} parity plot {shot_size} shot {random_state_adjust}")
                plot_parity(y_true_test_par, y_pred_test_par, flag_verbose, savedir, f"Task Test set {comp_name} parity plot {shot_size} shot {random_state_adjust}")
    
            shot_sizes_expres[shot_size_i, :] = [rmse_test, r2_test, mae_test, mare_test]
        
        res_tasks_np.append([
            comp_name, 
            rmse_test_full, r2_test_full, mae_test_full, mare_test_full,
            rmse_test_zero, r2_test_zero, mae_test_zero, mare_test_zero,
            *shot_sizes_expres.flatten(), 
            rmse_test_full_aft, r2_test_full_aft, mae_test_full_aft, mare_test_full_aft,
            similarity_vals[similarity_vals.index == comp_name]['max'].values[0],
            random_state_adjust
        ])

    return res_tasks_np

def collate_fn_support(batch, support_ratio=0.0, support_size=128):
    global current_epoch
    dataL_list = [pair[0] for pair in batch]
    dataS_list = [pair[1] for pair in batch]

    rng = random.Random(current_epoch)
    indices = list(range(len(batch)))
    rng.shuffle(indices)

    n_total = len(indices)
    min_set_size = 4

    max_support_size = max(0, n_total - min_set_size)

    if support_size is not None:
        support_size = min(support_size, max_support_size)
        split = support_size
    else:
        split = int(support_ratio * n_total)
        split = min(split, max_support_size)

    if split < min_set_size or split >= n_total:
        support_idx = indices
        query_idx = indices
    else:
        support_idx, query_idx = indices[:split], indices[split:]

    dataL_list_support = [dataL_list[i] for i in support_idx]
    dataL_list_query   = [dataL_list[i] for i in query_idx]
    dataS_list_support = [dataS_list[i] for i in support_idx]
    dataS_list_query   = [dataS_list[i] for i in query_idx]

    return (Batch.from_data_list(dataL_list_support),
            Batch.from_data_list(dataS_list_support),
            Batch.from_data_list(dataL_list_query),
            Batch.from_data_list(dataS_list_query))

def get_loaders(df, mode, batch_size=8192):
    if mode == 'pre-training':
        zero_rows_split = df[df['split'] == 0].index
        num_to_change_split_val = int(np.ceil(len(zero_rows_split) * 0.1))
        rows_to_change_split_val = np.random.choice(zero_rows_split, num_to_change_split_val, replace=False)
        df.loc[rows_to_change_split_val, 'split'] = 6

        train_sampler = TaskBatchSampler(df[df['split'] == 0]['graphs'].values)
        train_loader = DataLoader(df[df['split'] == 0]['graphs'].values, batch_sampler=train_sampler, collate_fn=collate_fn_support)

        meta_valid_loader = PYGDataLoader(df[df['split'] == 6]['graphs'].values, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        full_test_loader = PYGDataLoader(df[df['split'].isin([1,-1])]['graphs'].values, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        return train_loader, meta_valid_loader, full_test_loader
    elif mode == 'fine-tuning':
        adapt_loader = PYGDataLoader(df[df['split'] == 2]['graphs'].values, batch_size=batch_size, shuffle=True)
        valid_loader = PYGDataLoader(df[df['split'] == 3]['graphs'].values, batch_size=batch_size, shuffle=True)
        test_loader = PYGDataLoader(df[df['split'] == -1]['graphs'].values, batch_size=batch_size, shuffle=True)
        full_test_loader = PYGDataLoader(df[df['split'].isin([1,-1])]['graphs'].values, batch_size=batch_size, shuffle=True)

        return adapt_loader, valid_loader, test_loader, full_test_loader

def execute_finetuning_experiments(df_full, 
                                   maml, solute_name_dict, use_cross_val, use_adjust_val_subset, fold_to_use, 
                                   compounds_names_to_test, similarity_vals,
                                   input_channels, cond_names, savedir, device, scalers, flag_scale_target, flag_two_transforms,
                                   validation_schema='random', 
                                   adapt_selection_method='centroids',
                                   epochs = 25,
                                   flag_verbose=False,flag_no_ft=False,
                                   test_all_sets=False,train_loader=None,valid_loader=None,
                                   shot_sizes = [8, 16, 32, 64, 128],
                                   random_state_adjust_list = [40,41,42,43,44],
                                  ):
    res_tasks = []
    for random_state_adjust in random_state_adjust_list:
        res_tasks_np_temp = fine_tuning_experiment_run(df_full, random_state_adjust,
                                                       maml, solute_name_dict, use_cross_val, use_adjust_val_subset, fold_to_use, 
                                                       compounds_names_to_test, similarity_vals,
                                                       input_channels, cond_names, savedir, device, scalers, flag_scale_target, flag_two_transforms,
                                                       validation_schema=validation_schema, 
                                                       adapt_selection_method=adapt_selection_method, epochs = epochs,
                                                       flag_verbose=flag_verbose,shot_sizes=shot_sizes,flag_no_ft=flag_no_ft,
                                                       test_all_sets=test_all_sets,train_loader=train_loader,valid_loader=valid_loader,)
        if not res_tasks_np_temp: 
            print('Fine-tuning not applicable since there are no testing tasks specified')
            return None
        res_tasks.append(res_tasks_np_temp)
    res_tasks_np = np.vstack(res_tasks)
    shot_sizes_names = [f"{metric}-{shot_size}-shot" for shot_size in shot_sizes for metric in ['RMSE', 'R2', 'MAE', 'MARE']]

    res_tasks_df = pd.DataFrame(data=res_tasks_np, 
                                columns=['compound', 
                                         'RMSE-full-test-before-ft', 'R2-full-test-before-ft', 'MAE-full-test-before-ft', 'MARE-full-test-before-ft', 
                                         'RMSE-zero-shot', 'R2-zero-shot','MAE-zero-shot','MARE-zero-shot',
                                         *shot_sizes_names, 
                                         'RMSE-full-test', 'R2-full-test', 'MAE-full-test', 'MARE-full-test',
                                         'max_similarity', 'seed'])
    return res_tasks_df

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_everything(42)
    torch.backends.cudnn.benchmark = True

    # Configuration
    act = 'activity'
    target_column = 'lngamma' # 'IDAC (exptl)', 'lngamma'
    flag_scale_target = True
    flag_cleaning = False
    flag_verbose = 1
    if flag_scale_target:
        flag_two_transforms = False
        flag_transformation = 'quantile_normal'
    savedir = '.'
    flag_just_fine_tune = False
    flag_no_ft = False

    if not os.path.exists(savedir):
        os.makedirs(savedir)
        print("LOG | Created a directory to store results")

    fold_to_use = 0
    validation_schema = 'smiles'  # Options: 'random', 'smiles', 'cations', 'anions'
    use_adjust_val_subset = True
    use_cross_val = False

    df, scalers = get_df(act, target_column, flag_verbose, flag_cleaning, flag_scale_target, flag_transformation, flag_two_transforms)
    solute_name_dict, substances_iupac_list, similarity_vals, compounds_names_to_test = get_molecules_similarity(df, flag_verbose, savedir)
    df.drop('solutes', axis=1, inplace=True)
    df = prepare_splits_pre_training(df, compounds_names_to_test=compounds_names_to_test, solute_name_dict=solute_name_dict,)
    df, tasks, tasks_dict, y = extract_tasks_and_targets(df)
    df = add_graphs_to_df(df, tasks)
    train_loader, meta_valid_loader, full_test_loader = get_loaders(df, 'pre-training', batch_size=8192)
        
    cond_names = ['temps']
    input_channels = df['graphs'].values[0][0].x.shape[1]

    GCN_model = GCN(
        AtomMessagePassing,
        input_channels = input_channels, input_channels2 = int(input_channels / 2),
        embedding_size = 128, hidden_dim = 256,
        linear_size = [256,128], add_params_num = len(cond_names))

    maml = l2l.algorithms.MAML(GCN_model, lr=5e-5, first_order=True).to(device)
    optimizer_maml = torch.optim.Adam(maml.parameters(), lr=5e-6)

    criterion = torch.nn.MSELoss()
    global current_epoch
    current_epoch = 0

    if not flag_just_fine_tune:
        maml, train_losses, valid_losses, r2s_valid = main_maml_training_loop(maml, 
                                                                        train_loader, meta_valid_loader, scalers,
                                                                        optimizer_maml, cond_names, criterion, input_channels, flag_scale_target, flag_two_transforms,
                                                                        device=device, savedir=savedir,
                                                                        epochs=30000, adaptation_steps=3, meta_batch_size=32,
                                                                        flag_clean_start=True
                                                                            )


    if flag_verbose and not flag_just_fine_tune:
        plot_train_losses_vs_epoch(train_losses, valid_losses, r2s_valid, savedir, flag_verbose)
        plot_r2_scores_pretraining(train_losses, valid_losses, r2s_valid, savedir, flag_verbose)

    res_tasks_df = execute_finetuning_experiments(df, 
                                                  maml, solute_name_dict, use_cross_val, use_adjust_val_subset, fold_to_use, 
                                                  compounds_names_to_test, similarity_vals,
                                                  input_channels, cond_names, savedir, device, scalers, flag_scale_target, flag_two_transforms,
                                                  validation_schema='smiles', adapt_selection_method='centroids',
                                                  epochs = 30, flag_verbose=flag_verbose, flag_no_ft=flag_no_ft, 
                                                  test_all_sets=True,train_loader=train_loader,valid_loader=meta_valid_loader,
                                                  shot_sizes = [8,16,32,64,128],
                                                  random_state_adjust_list = [40,41,42,43,44],
                                                )
    res_tasks_df

    if not flag_no_ft:
        res_tasks_df_sorted = res_tasks_df.sort_values(by=['compound', 'seed'])
        res_tasks_df_sorted.to_excel(f'{savedir}/random-upto1024.xlsx')
        res_tasks_df_sorted

if __name__ == '__main__':
    main()