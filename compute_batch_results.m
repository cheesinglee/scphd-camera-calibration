clear_custom

results_dir = 'results/disparity_sim3_triangulation_only' ;
groundtruth_file = 'data/ground_truth_3.mat' ;
load(groundtruth_file) ;
n_runs = 10 ;
n_steps = numel(ground_truth) ;

results = cell(1,n_runs) ;
n_runs_true = n_runs ;
for n = 1:n_runs
    dirname = [results_dir,filesep,num2str(n),filesep] ;
    try
        results{n} = load_disparity_results(dirname,groundtruth_file) ;
    catch
        results{n} = [] ;
        n_runs_true = n_runs_true - 1 ;
    end
end

%%
mean_camerror = zeros(6,n_steps) ;
mean_camcov = zeros(6,6,n_steps) ;
mean_ospa1 = zeros(3,n_steps) ;
mean_ospa2 = zeros(3,n_steps) ;
for n = 1:n_runs 
    disp(n) ;
%     plot(results{n}.cam_error(1,:)) ;
    if ~isempty(results{n})
        mean_camerror = mean_camerror + results{n}.cam_error ;
        mean_camcov = mean_camcov + results{n}.cov_traj([1:3,7:9],[1:3,7:9],:) ;
        mean_ospa1 = mean_ospa1 + results{n}.ospa1 ;
        mean_ospa2 = mean_ospa2 + results{n}.ospa2 ;
    end
end

mean_camerror = mean_camerror/n_runs_true ;
mean_ospa1 = mean_ospa1/n_runs_true ;
mean_ospa2 = mean_ospa2/n_runs_true ;

figure(1)
set(gca,'fontsize',16,'fontweight','bold') 
plot(mean_ospa1(1,:))
grid on
ylim([0,1])
xlabel 'Time Step'
ylabel 'OSPA Error'
export_fig 'ospa_triangulation.pdf' -transparent
% save 'batch_results.mat'