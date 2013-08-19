function results = load_disparity_results( results_dir, ground_truth )
%COMPUTE_DISPARITY_ERROR Summary of this function goes here
%   Detailed explanation goes here
    load(ground_truth) ;
    n_steps = size(true_traj,2) ;
    
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
        n = sum(feature_weights >= 0.33) ;
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
    
    %% calculate camera error
    cam_error = exp_traj([1:3,7:9],:) - repmat(campose_2',1,n_steps) ;
    
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

        [dist,loc,cn] = ospa_dist(mean1,mean_true1,10,1) ;
    %         'DistanceMetric','Hellinger',...
    %         'HellingerCovariances',{cov1,cov_true1}) ;
        ospa1(:,k) = [dist;loc;cn] ;

        [dist,loc,cn] = ospa_dist(mean2,mean_true2,10,1) ;
    %         'DistanceMetric','Hellinger',...
    %         'HellingerCovariances',{cov2,cov_true2}) ;
        ospa2(:,k) = [dist;loc;cn] ;
    end
    
    results.exp_traj = exp_traj ;
    results.cov_traj = cov_traj ;
    results.particle_poses = particle_poses ;
    results.particle_weights = particle_weights ;
    results.maps = maps ;
    results.maps_cam1 = maps_cam1 ;
    results.maps_cam2 = maps_cam2 ;
    results.cardinality = cardinality ;
    results.cam_error = cam_error ;
    results.ospa1 = ospa1 ;
    results.ospa2 = ospa2 ;
end

