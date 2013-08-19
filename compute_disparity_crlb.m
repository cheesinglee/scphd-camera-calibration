clear_custom
load 'data/disparity_sim6.mat'

n_particles = 256 ;
n_targets = size(true_traj,3) ;

features_disparity1 = GaussianMixture(6) ;
features_disparity1.means = nan(6,n_targets) ;
features_disparity1.covs = nan(6,6,n_targets) ;
features_disparity1.weights = ones(1,n_targets) ;

features_disparity2 = GaussianMixture(6) ;
features_disparity2.means = nan(6,n_targets) ;
features_disparity2.covs = nan(6,6,n_targets) ;
features_disparity2.weights = ones(1,n_targets) ;

features_3d = zeros(6,n_particles,n_targets) ;

nu = 3 ;
nv = 3 ;
d0 = 8 ;

birth_cov = diag([nu,nv,3,0.0001,0.0001,0.0001]).^2 ;

H = [eye(2),zeros(2,4)] ;
R = diag([nu,nv].^2) ;

ground_truth = cell(1,n_steps) ;

for k = 1:n_steps
    campose_2 = cam_traj(:,k) ;
    if mod(k,2) == 1
%         campose = campose_1 ;
        campose_old = campose_2 ;
    else
%         campose = campose_2 ;
        campose_old = campose_1 ;
    end
    
    % prediction
    if k > 1
        if mod(k,2) ~= 0
            particles_disparity = features_disparity2.sample(n_particles) ;
        else
            particles_disparity = features_disparity1.sample(n_particles) ;
        end
        particles_world = zeros(6,n_particles,n_targets) ;
        for i = 1:n_targets
            particles_3d = measurement_model.invertMeasurement(particles_disparity(:,:,i),campose_old) ;
            particles_3d = motion_model.computeNoisyMotion(particles_3d) ;
            
            particles_d = measurement_model.computeMeasurement(particles_3d,campose_1) ;
            m = mean(particles_d,2) ;
            c = cov(particles_d') ;
            features_disparity1.means(:,i) = m ;
            features_disparity1.covs(:,:,i) = c ;
            
            particles_d = measurement_model.computeMeasurement(particles_3d,campose_2) ;
            m = mean(particles_d,2) ;
            c = cov(particles_d') ;
            features_disparity2.means(:,i) = m ;
            features_disparity2.covs(:,:,i) = c ;
            
            particles_world(:,:,i) = particles_3d ;
        end
%         particles_world = reshape(particles_world,6,[]) ;
    end
    
    targets_k = reshape(true_traj(:,k,:),[6,n_targets]) ;
    in_range1 = measurement_model.checkInRange(targets_k,campose_1) ;
    in_range2 = measurement_model.checkInRange(targets_k,campose_2) ;
    
    covs1 = features_disparity1.covs ;
    means1 = features_disparity1.means ;
    for i = 1:n_targets
        if in_range1(i)
            P = covs1(:,:,i) ;
            means1(:,i) = measurement_model.computeMeasurement(targets_k(:,i),campose_1) ;
            % seen for first time
            if any(isnan(P))                
                covs1(:,:,i) = birth_cov ;
                means1(3,i) = d0 ;
            % updating with new measurement
            else
                S = H*P*H' + R ;
                K = P*H'/S ;
                covs1(:,:,i) = (eye(6) - K*H)*P  ;
            end
        end
    end
    features_disparity1.means = means1 ;
    features_disparity1.covs = covs1 ;
    
    covs2 = features_disparity2.covs ;
    means2 = features_disparity2.means ;
    for i = 1:n_targets
        if in_range2(i)
            P = covs2(:,:,i) ;
            means2(:,i) = measurement_model.computeMeasurement(targets_k(:,i),campose_2) ;
            % seen for first time
            if any(isnan(P))                
                covs2(:,:,i) = birth_cov ;
            % updating with new measurement
            else
                S = H*P*H' + R ;
                K = P*H'/S ;
                covs2(:,:,i) = (eye(6) - K*H)*P  ;
            end
        end
    end
    features_disparity2.means = means2 ;
    features_disparity2.covs = covs2 ;
    

    gt1 = features_disparity1.copy() ;
    mask1 = isnan(means1(1,:)) | ~in_range1 ;

    gt2 = features_disparity2.copy() ;
    mask2 = isnan(means2(1,:)) | ~in_range2 ;
    
    gt1.means(:,mask1) = [] ;
    gt1.covs(:,:,mask1) = [] ;
    gt1.weights(mask1) = [] ;
    
    gt2.means(:,mask2) = [] ;
    gt2.covs(:,:,mask2) = [] ;
    gt2.weights(mask2) = [] ;
    
    ground_truth{k}.gaussians(1) = gt1 ;
    ground_truth{k}.gaussians(2) = gt2 ;
    
    if (k > 1)
        particles_world(:,:,mask1) = [] ;
        ground_truth{k}.particles = reshape(particles_world,6,[]);
    else
        ground_truth{k}.particles = [nan;nan;nan] ;
    end
    
    ground_truth{k}.targets = targets_k ;
end

save('data/ground_truth_6.mat') ;

%% plot
close all
figure(1)
h_z1 = plot(nan,nan,'r+','markersize',12) ;
hold on
h_crlb1 = plot(nan,nan,'linewidth',2) ;
grid on
xlim([-100,900]) 
ylim([-100,700]) 
set(gca,'ydir','reverse')

figure(2)
h_z2 = plot(nan,nan,'r+','markersize',12) ;
hold on
h_crlb2= plot(nan,nan,'linewidth',2) ;
grid on
xlim([-100,900]) 
ylim([-100,700]) 
set(gca,'ydir','reverse')

figure(3)
h_3d = plot(nan,nan,'.') ;
hold on
h_3d_true = plot3(nan,nan,nan,'r+') ;
grid on
xlim([-66,66])
ylim([-66,66])
zlim([-10,200])
view(3)

tilefigs() ;

for k = 1:n_steps
    title(k)
    z1 = Z1{k} ;
    z2 = Z2{k} ;
    set(h_z1,'xdata',z1(1,:),'ydata',z1(2,:)) ;
    set(h_z2,'xdata',z2(1,:),'ydata',z2(2,:)) ;
    
    gt1 = ground_truth{k}.gaussians(1) ;
    gt2 = ground_truth{k}.gaussians(2) ;
    particles = ground_truth{k}.particles ;
    pp1 = gt1.plot_ellipses() ;
    pp2 = gt2.plot_ellipses() ;
    
    targets = ground_truth{k}.targets ;

    set(h_crlb1,'xdata',pp1(1,:),'ydata',pp1(2,:)) ;
    set(h_crlb2,'xdata',pp2(1,:),'ydata',pp2(2,:)) ;

    set(h_3d,'xdata',particles(1,:),'ydata',particles(2,:),'zdata',particles(3,:)) ;
    set(h_3d_true,'xdata', targets(1,:),'ydata',targets(2,:),'zdata',targets(3,:)) ;
%     pause(0.1)
    drawnow
end
