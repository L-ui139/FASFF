from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.check_dir = '/models'
    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/root/shared-nvme/LasHeR0327/got10k_lmdb'
    settings.got10k_path = '/root/shared-nvme/LasHeR0327/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/root/shared-nvme/LasHeR0327/itb'
    settings.lasot_extension_subset_path_path = '/root/shared-nvme/LasHeR0327/lasot_extension_subset'
    settings.lasot_lmdb_path = '/root/shared-nvme/LasHeR0327/lasot_lmdb'
    settings.lasot_path = '/root/shared-nvme/LasHeR0327/lasot'
    settings.network_path = '/root/miniconda3/code/TBSI/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/root/shared-nvme/LasHeR0327/nfs'
    settings.otb_path = '/root/shared-nvme/LasHeR0327/otb'
    settings.prj_dir = '/root/miniconda3/code/TBSI'
    settings.result_plot_path = '/root/miniconda3/code/TBSI/output/test/result_plots'
    settings.results_path = '/root/miniconda3/code/TBSI/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/root/miniconda3/code/TBSI/output'
    settings.segmentation_path = '/root/miniconda3/code/TBSI/output/test/segmentation_results'
    settings.tc128_path = '/root/shared-nvme/LasHeR0327/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/root/shared-nvme/LasHeR0327/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/shared-nvme/LasHeR0327/trackingnet'
    settings.uav_path = '/root/shared-nvme/LasHeR0327/uav'
    settings.vot18_path = '/root/shared-nvme/LasHeR0327/vot2018'
    settings.vot22_path = '/root/shared-nvme/LasHeR0327/vot2022'
    settings.vot_path = '/root/shared-nvme/LasHeR0327/VOT2019'
    settings.youtubevos_dir = ''
    settings.lasher_path = '/root/shared-nvme/LasHeR0327/lasher'
    settings.rgbt234_path = '/root/shared-nvme/RGBT234/rgbt234'
    settings.gtot_path = '/root/shared-nvme/GTOT/gtot'

    return settings

