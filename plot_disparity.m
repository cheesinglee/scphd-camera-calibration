%%
clear_custom
max_steps =  100;
% load('data/fov_test.mat')
load('data/disparity_sim.mat')
n_steps = min(n_steps,max_steps) ;
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
particle_weights = cell(n_steps) ;
particle_poses = cell(n_steps) ;
disp('load data: ') ;
for k = 1:n_steps
    filename = [num2str(k-1),'.mat'] ;
    disp(filename)
    load(filename)
    w = weights ;
    p = particles ;
    
    weighted_particles = repmat(w',12,1).*p ;
    exp_traj(:,k) = sum(weighted_particles,2) ;
    cov_traj(:,:,k) = weightedcov(particles',w') ; 
    
    feature_weights = features.weights ;
%     n = min(round(sum(feature_weights)),size(features.particles,2)/150) ;
    n = sum(feature_weights >= 0.4) ;
    
    maps{k} = features.particles(:,1:100*n)  ;
    particle_weights{k} = w ;
    particle_poses{k} = p ;
end
%%
close all
figure(1)
h = plot3(0,0,0,'.') ;
hold on
hh = plot3(0,0,0,'r+','markersize',10) ;
h_traj = plot3(nan,nan,nan,'k--') ;
grid on

h_axes1 = measurement_model.drawAxes(campose_1,10) ;
h_axes2 = measurement_model.drawAxes(campose_2,10) ;
set(h_axes2,'Color','r')
xlim([-100,100])
ylim([-100,100]) 
zlim([-50,500]) ;
axis square

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
view(3)

figure(4)
h_weights = plot(nan,nan,'.') ;
hold on
grid on
ylim([0,1]) ;
xlim([-6,3]) ;

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
        
        p_image = measurement_model.computeMeasurement(p(1:3,:),cam_traj(1:6,k)) ; 
        set(ha,'xdata',p_image(1,:),'ydata',p_image(2,:),'zdata',p_image(3,:)) ;
        zk = Z2{k} ;
        set(hb,'xdata',zk(1,:),'ydata',zk(2,:)) ;
    end
    
    set(h_weights,'xdata',particle_poses{k}(3,:),'ydata',particle_weights{k}) ;
    drawnow
    pause(0.1)
end