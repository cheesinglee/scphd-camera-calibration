clear_custom
load 'data/disparity_sim.mat'

n_particles = 100 ;
n_targets = size(true_traj,3) ;

features_disparity = GaussianMixture(6) ;
features_disparity.means = nan(6,n_targets) ;
features_disparity.covs = nan(6,6,n_targets) ;
features_disparity.weights = ones(1,n_targets) ;
features_3d = zeros(6,n_particles,n_targets) ;

birth_cov = diag([10,10,4,6,6,0.5]).^2 ;

H = [eye(2),zeros(2,4)] ;
R = 10*eye(2) ;

ground_truth = cell(1,n_steps) ;

for k = 1:n_steps
    if mod(k,2) == 1
        campose = campose_1 ;
        campose_old = campose_2 ;
    else
        campose = campose_2 ;
        campose_old = campose_1 ;
    end
    
    % prediction
    if k > 1
        particles_disparity = features_disparity.sample(n_particles) ;
        particles_world = zeros(6,n_particles,n_targets) ;
        for i = 1:n_targets
            particles_3d = measurement_model.invertMeasurement(particles_disparity(:,:,i),campose_old) ;
            particles_3d = motion_model.computeNoisyMotion(particles_3d) ;
            particles_d = measurement_model.computeMeasurement(particles_3d,campose) ;
            m = mean(particles_d,2) ;
            c = cov(particles_d') ;
            features_disparity.means(:,i) = m ;
            features_disparity.covs(:,:,i) = c ;
            particles_world(:,:,i) = particles_3d ;
        end
%         particles_world = reshape(particles_world,6,[]) ;
    end
    
    targets_k = reshape(true_traj(:,k,:),[6,n_targets]) ;
    in_range = measurement_model.checkInRange(targets_k,campose) ;
    
    covs = features_disparity.covs ;
    means = features_disparity.means ;
    for i = 1:n_targets
        if in_range(i)
            P = covs(:,:,i) ;
            means(:,i) = measurement_model.computeMeasurement(targets_k(:,i),campose) ;
            % seen for first time
            if any(isnan(P))                
                covs(:,:,i) = birth_cov ;
            % updating with new measurement
            else
                S = H*P*H' + R ;
                K = P*H'/S ;
                covs(:,:,i) = (eye(6) - K*H)*P  ;
            end
        end
    end
    features_disparity.means = means ;
    features_disparity.covs = covs ;
    
    gt = features_disparity.copy() ;
    mask = isnan(means(1,:)) | ~in_range ;
    gt.means(:,mask) = [] ;
    gt.covs(:,:,mask) = [] ;
    gt.weights(mask) = [] ;
    ground_truth{k}.gaussians = gt ;
    if (k > 1)
        particles_world(:,:,mask) = [] ;
        ground_truth{k}.particles = reshape(particles_world,6,[]);
    else
        ground_truth{k}.particles = [nan;nan;nan] ;
    end
end

save('ground_truth.mat') ;

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
    
    gt = ground_truth{k}.gaussians ;
    particles = ground_truth{k}.particles ;
    pp = gt.plot_ellipses() ;
    if mod(k,2) == 1
        set(h_crlb1,'xdata',pp(1,:),'ydata',pp(2,:)) ;
        set(h_crlb2,'xdata',nan,'ydata',nan) ;
    else
        set(h_crlb2,'xdata',pp(1,:),'ydata',pp(2,:)) ;
        set(h_crlb1,'xdata',nan,'ydata',nan) ;
    end
    
    set(h_3d,'xdata',particles(1,:),'ydata',particles(2,:),'zdata',particles(3,:))
    pause(0.2)
    drawnow
end
