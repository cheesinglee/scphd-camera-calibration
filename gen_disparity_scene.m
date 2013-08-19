function gen_disparity_scene()
    clear_custom
    filename = 'data/disparity_sim6.mat' ;
    if exist(filename,'file') 
        in = input([filename,' already exists. Proceed? [y/N] '],'s') ;
        if isempty(in) || lower(in)  ~= 'y'
            return
        end
    end
    always_visible = true ;
    % measurement model
    lambda = 10 ;
    pd = 0.95 ;
    measurement_model = DisparityMeasurementModel(400,300,-895.6561,-891.2656,...
        'B',1,'nu',3,'nv',3,'ClutterRate',lambda,'pd',pd) ;

    % extrinsics
    % x2 = 30 ; y2 = 20 ; z2 = -10 ;
    x2 = 0 ; y2 = -64 ; z2 = 0 ;
    % theta2 = -pi/8 ; phi2 = 0 ; psi2 = pi/6 ;
    theta2 = -pi/8 ; phi2 = 0 ; psi2 = 0 ;

    campose_1 = zeros(6,1) ;
%     campose_2 = [x2,y2,z2,theta2,phi2,psi2]' ;
    campose_2 = campose_1 ;

    n_steps = 300 ;
    dt = 0.5 ;

    %% camera motion
    std_ax_cam = 0.05 ;
    std_ay_cam = 0.05 ;
    std_az_cam = 0.05 ;
    std_atheta_cam = eps ;
    std_aphi_cam = eps ;
    std_apsi_cam = eps ;
    Q_camera = diag([std_ax_cam,std_ay_cam,std_az_cam,std_atheta_cam,std_aphi_cam,std_apsi_cam].^2) ;
    camera_model = LinearCV3DMotionModel(Q_camera,dt) ;

    vx0 = 1.0 ;
    vy0 = -0.0 ;
    vz0 = 0.0 ;
    vtheta0 = 0.0 ;
    vphi0 = 0 ;
    vpsi0 = 0.0 ;

    X0 = [campose_2;vx0;vy0;vz0;vtheta0;vphi0;vpsi0] ;
    cam_traj = repmat(X0,1,n_steps) ;
    t = 0:n_steps-1 ;
    % cam_x = 50*sin(pi/50*t) ;
    % cam_y = -50*cos(pi/50*t) ;
    % cam_z = t/n_steps*50 ;
    % cam_theta = pi/8*cos(pi/50*t) ;
    % cam_phi = pi/8*sin(pi/50*t) ;
    cam_x = -30*sin(pi/n_steps*t) ;
    cam_y = zeros(1,n_steps) ;
    cam_z = zeros(1,n_steps) ;
%     cam_z = 30*sin(pi/n_steps*t) ;
    
    cam_theta = zeros(1,n_steps) ;
    cam_phi = pi/6*sin(pi/n_steps*t) ;
    cam_psi = zeros(1,n_steps) ;
    cam_traj = [cam_x;cam_y;cam_z;cam_theta;cam_phi;cam_psi] ;
    % for k = 2:n_steps
    %     cam_traj(:,k) = camera_model.computeNoisyMotion(cam_traj(:,k-1)) ;
    % end

    %% make target trajectories
    n_targets = 10 ;
    xmin = -30 ;
    xmax = 30 ;
    ymin = -30 ;
    ymax = 30 ;
    zmin = 50 ;
    zmax = 200 ;
    region = [[xmin,ymin,zmin] ; [xmax,ymax,zmax]];
    std_v = [1.5,1.5,2.0] ;
    
    % target motion model
%     std_ax = 0.15 ;
%     std_ay = 0.15 ;
%     std_az = 0.15 ;
    std_ax = eps ;
    std_ay = eps ;
    std_az = eps ;
    Q = diag([std_ax,std_ay,std_az]).^2 ;
    motion_model = LinearCV3DMotionModel(Q,dt) ;
    
    good_trajectory = false ;
    while (~good_trajectory)
%         true_traj = makeTargetTrajectories(n_targets,n_targets,region,...
%             std_v,motion_model,n_steps) ;
        
        % stationary targets
        x0 = rand(1,n_targets).*(xmax-xmin) + xmin ;
        y0 = rand(1,n_targets).*(ymax-ymin) + ymin ;
        z0 = rand(1,n_targets).*(zmax-zmin) + zmin ;
        v0 = zeros(1,n_targets) ;
        true_traj = reshape([x0;y0;z0;v0;v0;v0],6,1,n_targets) ;
        true_traj = repmat(true_traj,[1,n_steps,1]) ;

        good_trajectory = true ;
        %% make measurements
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
    end
    %% export data
    save(filename) ;

    % %% write configuration
    % fid = fopen('disparity_sim.cfg','w') ;
    % fprintf(fid,'#prior distribution for calibration parameters\n') ;
    % fprintf(fid,'prior : {\n') ;
    % fprintf(fid,'\tcartesian : {\n') ;
    % fprintf(fid,'\t\tx : {mean = %f ; std = %f}\n',campose_2(1),0.0) ;
    % fprintf(fid,'\t\ty : {mean = %f ; std = %f}\n',campose_2(1),0.0) ;
    % fprintf(fid,'\t\tz : {mean = %f ; std = %f}\n',campose_2(1),0.0) ;

    %% plot
    f1 = figure(1) ;
    clf
    subplot(2,2,[1,3]) ;
    h_axes1 = measurement_model.drawAxes(campose_1,20,f1) ;
    hold on
    h_axes2 = measurement_model.drawAxes(campose_2,20,f1) ;
    set(h_axes2,'Color','r')

    img_plane1 = measurement_model.drawImagePlane(campose_1,20) ;
    img_plane2 = measurement_model.drawImagePlane(campose_2,20) ;

    h_plane1 = plot3(img_plane1(1,:),img_plane1(2,:),img_plane1(3,:))
    h_plane2 = plot3(img_plane2(1,:),img_plane2(2,:),img_plane2(3,:),'r')

    axis equal

    % x_all = reshape(true_traj(1,:,:),1,[]) ;
    % y_all = reshape(true_traj(2,:,:),1,[]) ;
    % z_all = reshape(true_traj(3,:,:),1,[]) ;
    % xlim([min(x_all),max(x_all)])
    % ylim([min(y_all),max(y_all)])
    % zlim([min(z_all),max(z_all)])
    xlim([xmin,xmax])
    ylim([ymin,ymax])
    zlim([-20,zmax])

    x = squeeze(true_traj(1,1,:)) ;
    y = squeeze(true_traj(2,1,:)) ;
    z = squeeze(true_traj(3,1,:)) ;

    h_targets = plot3(x,y,z,'+') ;
%     view([0,0,-1])

    subplot(2,2,2)
    h_img1 = plot(nan,nan,'+') ;
    xlim([0,800])
    ylim([0,600])
    grid on
    set(gca,'ydir','reverse')

    subplot(2,2,4)
    h_img2 = plot(nan,nan,'r+') ;
    xlim([0,800])
    ylim([0,600])
    grid on
    set(gca,'ydir','reverse')

    for k = 1:n_steps
        subplot(2,2,[1,3])
        img_plane1 = measurement_model.drawImagePlane(campose_1,20) ;
        img_plane2 = measurement_model.drawImagePlane(cam_traj(1:6,k),20) ;

        set(h_plane1,'xdata',img_plane1(1,:),'ydata',img_plane1(2,:),'zdata',img_plane1(3,:)) ;
        set(h_plane2,'xdata',img_plane2(1,:),'ydata',img_plane2(2,:),'zdata',img_plane2(3,:)) ;
%         plot3(img_plane1(1,:),img_plane1(2,:),img_plane1(3,:))
%         plot3(img_plane2(1,:),img_plane2(2,:),img_plane2(3,:),'r')
%         measurement_model.drawAxes(campose_1,20,f1) ;
%         measurement_model.drawAxes(cam_traj(1:6,k),20,f1) ;
%         set(h_axes2,'Color','r')
        x = squeeze(true_traj(1,k,:)) ;
        y = squeeze(true_traj(2,k,:)) ;
        z = squeeze(true_traj(3,k,:)) ;
        set(h_targets,'xdata',x,'ydata',y,'zdata',z) ;
        axis equal
        grid on
    %     set(h_targets,'xdata',x,'ydata',y,'zdata',z) ;
        z1_k = Z1{k} ;
        z2_k = Z2{k} ;
        if numel(z1_k) > 0
            set(h_img1,'xdata',z1_k(1,:),'ydata',z1_k(2,:)) ;
        end

        if numel(z2_k) > 0
            set(h_img2,'xdata',z2_k(1,:),'ydata',z2_k(2,:)) ;
        end

        title(k)
    %     pause(0.05)
        drawnow
    end
end

function true_traj =  makeTargetTrajectories(n_targets, n_initial, region,std_v,...
                                        motion_model,n_steps)
%     n_targets = 7 ;
%     n_initial = 7 ;

    % observation region
    xmin = region(1,1) ;
    ymin = region(1,2) ;
    zmin = region(1,3) ;
    xmax = region(2,1) ;
    ymax = region(2,2) ;
    zmax = region(2,3) ;

    % birth velocities
    std_vx0 = 1.5 ;
    std_vy0 = 1.5 ;
    std_vz0 = 2.0 ;
    % std_vx0 = 0 ;
    % std_vy0 = 0 ;
    % std_vz0 = 0 ;

    % target birth states
    x0 = rand(1,n_targets).*(xmax-xmin) + xmin ;
    y0 = rand(1,n_targets).*(ymax-ymin) + ymin ;
    z0 = rand(1,n_targets).*(zmax-zmin) + zmin ;

    xk = rand(1,n_targets).*(xmax-xmin) + xmin ;
    yk = rand(1,n_targets).*(ymax-ymin) + ymin ;
    zk = rand(1,n_targets).*(zmax-zmin) + zmin ;
    vx0 = randn(1,n_targets)*std_vx0 ;
    vy0 = randn(1,n_targets)*std_vy0 ;
    vz0 = randn(1,n_targets)*std_vz0 ;
    birth_states = [x0;y0;z0;vx0;vy0;vz0] ;
    end_states = [xk;yk;zk] ;

    % target appearance times (all within first half of scenario)
    birth_times = floor(rand(1,n_targets)*n_steps/2) + 1 ;
    birth_times(1:n_initial) = 1 ;

    true_traj = nan(6,n_steps,n_targets) ;
    for i = 1:n_targets
        t0 = birth_times(i) ;
        true_traj(:,t0,i) = birth_states(:,i) ;
        n = n_steps - t0 + 1 ;
        for j = 1:3
            true_traj(j,t0:n_steps,i) = linspace(birth_states(j,i),end_states(j,i),n) ;
            true_traj(j+3,t0:n_steps,i) = (end_states(j,i) - birth_states(j,i))/n ;
        end
    %     for k = t0:n_steps-1
    %         true_traj(:,k+1,i) = motion_model.computeNoisyMotion(true_traj(:,k,i)) ;
    %     end
    end
end