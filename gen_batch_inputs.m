clear_custom
filename = 'data/disparity_sim4.mat' ;
n_runs = 100 ;

load(filename)

for n = 1:n_runs
    disp(n)
    Z1 = {} ; Z2 = {} ;
    for k = 1:n_steps
        target_states = reshape(true_traj(:,k,:),6,n_targets) ;
        idx_exist = ~isnan(target_states(1,:)) ;
        target_states = target_states(1:3,idx_exist) ;
        if numel(target_states) > 0
            Z1{k} = measurement_model.computeNoisyMeasurement(target_states,campose_1) ;
            Z2{k} = measurement_model.computeNoisyMeasurement(target_states,cam_traj(1:6,k)) ;

            if (always_visible)
                visible1 = measurement_model.checkInRange(target_states,campose_1) ;
                visible2 = measurement_model.checkInRange(target_states,cam_traj(1:6,k)) ;
                if (~all(visible1) || ~all(visible2))
                    good_trajectory = false ;
                    break
                end
            end
        else
            Z1{k} = [] ;
            Z2{k} = [] ;
        end       
    end
    batchdir = ['data/batch/',num2str(n)] ;
    mkdir(batchdir) ;
    batch_filename = [batchdir,filesep,'/data.mat']    ;
    save(batch_filename) ;
end

