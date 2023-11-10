from prettytable import PrettyTable

def make_deme_table(model_instance_obj):
    """Output table of demes and thetas"""
    deme_ids_list = list(sum(model_instance_obj.deme_ids, ()))
    deme_table = PrettyTable()
    deme_table.field_names = ["Deme ID", "Epoch", "Theta (block)"]
    for epoch_idx, epoch in enumerate(model_instance_obj.epochs):
        for deme_idx, deme in enumerate(epoch.deme_ids):
            deme_table.add_row([deme,
                                epoch_idx,
                                f'{epoch.thetas[deme_idx]:.3e}'])

    return deme_table


def make_migration_table(model_instance_obj):
    """Output table of migration rates"""
    mig_table = PrettyTable()
    mig_table.field_names = ["Source->Target", "Epoch", "Migration rate (M)"]


    for epoch_idx, epoch in enumerate(model_instance_obj.epochs):
        if epoch.migration is True:
            if epoch.migration_direction is not None:
                assert len([epoch.migration_sources]) == 1
                assert len([epoch.migration_targets]) == 1
                mig_table.add_row([f"{epoch.migration_sources}->{epoch.migration_targets}",
                                    epoch_idx,
                                    epoch.migration_rates[0]])
            elif epoch.asymmetric_migration is False:
                mig_table.add_row([f"{epoch.deme_ids[0]}<->{epoch.deme_ids[1]}",
                                    epoch_idx,
                                    epoch.migration_rates[0]])
            elif epoch.asymmetric_migration is True and epoch.migration_direction is None:
                mig_table.add_row([f"{epoch.deme_ids[0]}->{epoch.deme_ids[1]}",
                                    epoch_idx,
                                    epoch.migration_rates[0]])
                mig_table.add_row([f"{epoch.deme_ids[1]}->{epoch.deme_ids[0]}",
                                    epoch_idx,
                                    epoch.migration_rates[1]])
                    
    return mig_table


def make_epoch_table(model_instance_obj):
    """Output table of epoch data"""
    epoch_table = PrettyTable()
    epoch_table.field_names = ["Epoch",
                                "Start time (2N gen)",
                                "End time (2N gen)",
                                 "Demes",
                                    "Migration sources",
                                    "Migration targets",
                                        "Asymmetric migration"]
    for epoch_idx, epoch in enumerate(model_instance_obj.epochs):
            
        if epoch.end_time is not None:
            display_end = round(epoch.end_time, 3)
        else:
            display_end = None

        epoch_table.add_row(
            [
                epoch_idx,
                epoch.start_time,
                display_end,
                epoch.deme_ids,
                epoch.migration_sources,
                epoch.migration_targets,
                epoch.asymmetric_migration

            ]
        )

    return epoch_table


def print_output(model_instance_obj):
    print(f"""
        Model with {model_instance_obj.n_theta_params} demes and {len(model_instance_obj.epochs)} epochs.

        Negative log-likelihood: {model_instance_obj.negll}
        Composite likelihood AIC: {model_instance_obj.claic}

        Migration rate estimates: 

{make_migration_table(model_instance_obj)}

            Demes: 

{make_deme_table(model_instance_obj)}

            Epochs: 

{make_epoch_table(model_instance_obj)}

            """
        )