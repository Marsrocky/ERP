function [ output_args ] = read2ma( csidata )
%READ2AMPH Summary of this function goes here
%   Detailed explanation goes here
    load(csidata);
    vdata = reshape(csidata(:,1,:), 3, 56); % 56 subcarriers
    magnitude = abs(vdata);
    phase = angle(vdata);
    save('magangle.mat', 'magnitude', 'phase');
end

