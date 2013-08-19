%%
clear_custom
max_steps = 150;
draw_plots = true ;

load 'data/ground_truth_6.mat'
% load('data/fov_test.mat')
load('data/disparity_sim6.mat')
results_dir = './' ;
n_steps = min(n_steps,max_steps) ;
n_particles = 256 ;

% load('data/spiral')

% measurement model
% measurement_model = DisparityMeasurementModel(400,300,-895.6561,-891.2656,'B',1,'nu',3,'nv',3) ;

% extrinsics
% x2 = 30 ; y2 = 20 ; z2 = -10 ;
% theta2 = -pi/8 ; phi2 = 0 ; psi2 = pi/6 ;
% 
% campose_1 = zeros(6,1) ;
% campose_2 = [x2,y2,z2,theta2,phi2,psi2]' ;

exp_traj = zeros(12,n_steps) ;
cov_traj = zeros(12,12,n_steps) ;
maps = cell(1,n_steps) ;
cardinality = zeros(1,n_steps) ;

maps_cam1 = cell(1,n_steps) ;
maps_cam2 = cell(1,n_steps) ;
particle_weights = cell(n_steps) ;
particle_poses = cell(n_steps) ;
disp('load data: ') ;
for k = 1:n_steps
    filename = [results_dir,num2str(k-1),'.mat'] ;
    disp(filename)
    load(filename)
    w = weights ;
    p = particles ;
    
    weighted_particles = repmat(w',12,1).*p ;
    exp_traj(:,k) = sum(weighted_particles,2) ;
    cov_traj(:,:,k) = weightedcov(particles',w') ; 
    
    feature_weights = features.weights ;
%     n = min(round(sum(feature_weights)),size(features.particles,2)/150) ;
    n = sum(feature_weights >= 0.5) ;
    cardinality(k) = sum(feature_weights) ;
    
    maps{k} = features.particles(:,1:n_particles*n)  ;
    
    campose_2_k = [exp_traj(1:3), exp_traj(7:9)]' ;
    disparity1 = measurement_model.computeMeasurement(maps{k},campose_1) ;
    disparity2 = measurement_model.computeMeasurement(maps{k},campose_2_k) ;
    gm1 = GaussianMixture(6) ;
    gm2 = GaussianMixture(6) ;
    m1 = zeros(6,n) ;
    c1 = zeros(6,6,n) ;
    m2 = zeros(6,n) ;
    c2 = zeros(6,6,n) ;
    for i = 1:n
        idx = (1:n_particles) + (i-1)*n_particles ;
        particles_n = disparity1(:,idx) ;
        m1(:,i) = mean(particles_n,2) ;
        c1(:,:,i) = cov(particles_n') ;
        
        particles_n = disparity2(:,idx) ;
        m2(:,i) = mean(particles_n,2) ;
        c2(:,:,i) = cov(particles_n') ;
    end
    gm1.add_gaussians(ones(1,n),m1,c1) ;
    gm2.add_gaussians(ones(1,n),m2,c2) ;
    maps_cam1{k} = gm1 ;
    maps_cam2{k} = gm2 ;
    particle_weights{k} = w ;
    particle_poses{k} = p ;
end

%% calculate ospa errors
ospa1 = zeros(3,n_steps) ;
ospa2 = zeros(3,n_steps) ;
disp('computing OSPA distances...')
for k = 1:n_steps
    disp([num2str(k),'/',num2str(n_steps)])
    map1 = maps_cam1{k} ;
    map2 = maps_cam2{k} ;
    
    gt1 = ground_truth{k}.gaussians(1) ;
    gt2 = ground_truth{k}.gaussians(2) ;

    mean_true1 = gt1.means(1:3,:) ;
    mean_true2 = gt2.means(1:3,:) ;
    cov_true1 = gt1.covs(1:3,1:3,:) ;
    cov_true2 = gt2.covs(1:3,1:3,:) ;
    
    mean1 = map1.means(1:3,:) ;
    mean2 = map2.means(1:3,:) ;
    
    cov1 = map1.covs(1:3,1:3,:) ;
    cov2 = map2.covs(1:3,1:3,:) ;
    
    [dist,loc,cn] = ospa_dist(mean1,mean_true1,1,1,...
        'DistanceMetric','Hellinger',...
        'HellingerCovariances',{cov1,cov_true1}) ;
    ospa1(:,k) = [dist;loc;cn] ;
    
    [dist,loc,cn] = ospa_dist(mean2,mean_true2,1,1,...
        'DistanceMetric','Hellinger',...
        'HellingerCovariances',{cov2,cov_true2}) ;
    ospa2(:,k) = [dist;loc;cn] ;
end

%%
if draw_plots
    close all
    fig1 = figure(1) ;
    h = plot3(0,0,0,'.') ;
    hold on
    hh = plot3(0,0,0,'r+','markersize',10) ;
    h_traj = plot3(nan,nan,nan,'k--') ;
    grid on

    campose_2(4:6) = -campose_2(4:6) ;
    h_axes1 = measurement_model.drawAxes(campose_1,10,fig1) ;
    h_axes2 = measurement_model.drawAxes(campose_2,10,fig1) ;
    set(h_axes2,'Color','r')
    xlim([-100,100])
    ylim([-100,100]) 
    zlim([-50,500]) ;
%     axis equal

    figure(2) 
    hhh = plot(nan,nan,'.','markersize',1) ;
    hold on
    grid on
    hhhh = plot(nan,nan,'r+','markersize',10) ;
    ylim([-100,700])
    xlim([-100,900]) ;
    set(gca,'ydir','reverse')

    figure(3)
    ha = plot(nan,nan,'.','markersize',1) ;
    hold on 
    grid on
    hb = plot(nan,nan,'r+','markersize',10) ;
    ylim([-100,700])
    xlim([-100,900]) ;
    set(gca,'ydir','reverse')
    grid on
    % view(3)

    figure(4)
    subplot 211
    h_weights = plot(nan,nan,'.') ;
    hold on
    grid on
    ylim([0,1]) ;
    % xlim([-6,3]) ;
    xlim([-50,3]) ;
    subplot 212
    h_weights2 = plot(nan,nan,'.') ;
    hold on
    grid on
    ylim([0,1]) ;
    xlim([-1,1]) ;

    tilefigs()
    for k = 1:n_steps
        set(h_traj,'xdata',exp_traj(1,1:k),...
            'ydata',exp_traj(2,1:k),...
            'zdata',exp_traj(3,1:k)) ;

        p = maps{k} ;

        if numel(p) > 0
            set(h,'xdata',p(1,:),'ydata',p(2,:),'zdata',p(3,:)) ;
        else
            set(h,'xdata',[],'ydata',[],'zdata',[]) ;
        end

        pp = reshape(true_traj(:,k,:),6,[]) ;
        set(hh,'xdata',pp(1,:),'ydata',pp(2,:),'zdata',pp(3,:)) ;
        title(k-1)

        if numel(p) > 0
            p_image = measurement_model.computeMeasurement(p(1:3,:),campose_1) ; 
            set(hhh,'xdata',p_image(1,:),'ydata',p_image(2,:),'zdata',p_image(3,:)) ;
            zk = Z1{k} ;
            set(hhhh,'xdata',zk(1,:),'ydata',zk(2,:)) ;

            cam_traj_k = [exp_traj(1:3,k);exp_traj(7:9,k)] ;
            cam_traj_k(4:6) = cam_traj_k(4:6) ;
            p_image = measurement_model.computeMeasurement(p(1:3,:),cam_traj_k) ; 
            set(ha,'xdata',p_image(1,:),'ydata',p_image(2,:),'zdata',p_image(3,:)) ;
            zk = Z2{k} ;
            set(hb,'xdata',zk(1,:),'ydata',zk(2,:)) ;
        end

        set(h_weights,'xdata',particle_poses{k}(1,:),'ydata',particle_weights{k}) ;
        set(h_weights2,'xdata',particle_poses{k}(8,:),'ydata',particle_weights{k}) ;
        drawnow
%         pause
%         pause(0.2)
    end
end
%%
figure(6)
clf
subplot 221
plot(exp_traj(1,:))
hold on
sigma = sqrt(squeeze(cov_traj(1,1,:)))' ;
plot(exp_traj(1,:) + 3*sigma,'k--') ;
plot(exp_traj(1,:) - 3*sigma,'k--') ;
plot(cam_traj(1,1:n_steps),'r') ;
grid on


subplot 222
vx = diff(cam_traj(1,1:n_steps)/dt) ;
plot(exp_traj(4,:))
hold on
sigma = sqrt(squeeze(cov_traj(4,4,:)))' ;
plot(exp_traj(4,:) + 3*sigma,'k--') ;
plot(exp_traj(4,:) - 3*sigma,'k--') ;
plot(vx,'r') ;
grid on

subplot 223
plot(exp_traj(8,:))
hold on
sigma = sqrt(squeeze(cov_traj(8,8,:)))' ;
plot(exp_traj(8,:) + 3*sigma,'k--') ;
plot(exp_traj(8,:) - 3*sigma,'k--') ;
plot(cam_traj(5,1:n_steps),'r') ;
grid on

subplot 224
v_phi = diff(cam_traj(5,1:n_steps)/dt) ;
plot(exp_traj(11,:))
hold on
sigma = sqrt(squeeze(cov_traj(11,11,:)))' ;
plot(exp_traj(11,:) + 3*sigma,'k--') ;
plot(exp_traj(11,:) - 3*sigma,'k--') ;
plot(v_phi,'r') ;
grid on

%% plot ospa
figure
plot(ospa1(1,:))

