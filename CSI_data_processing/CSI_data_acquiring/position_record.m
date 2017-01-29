clear all;
close all;
no_position=6;
total_port={'8866', '8867'};
no_package=10;
packet_actual=1;

for i=1:no_position
    for j=1:length(total_port)
        for k=1:no_package
            fileName = strcat(int2str(i), '/', total_port(j), '-', int2str(k));
            read_log_file_be (fileName{1})
            Data = ans;
            
            %packet analysis
            packet1 = Data{1};
            if isempty(packet1)==0
                if isempty(packet1.csi)==0
                    if mean(squeeze(packet1.csi(1,1,:))')~=0
                        csidata = packet1.csi;
                        csi_rawdata{i}{j}{k} = squeeze(csidata(:,1,:));
                    end
                end
            end
        end
    end
end

save csi_rawdata csi_rawdata
% calculate raw mag and phase
for i=1:no_position
    for j=1:length(total_port)
        for k=1:no_package
            csi_mag{i}{j}{k}=db(csi_rawdata{i}{j}{k});
            csi_phase{i}{j}{k}=angle(csi_rawdata{i}{j}{k});
        end
    end
end

% reshape to vector
for i=1:no_position
    for j=1:length(total_port)
        temp = zeros(no_package, 342);
        temp2 = zeros(no_package, 342);
        for k=1:no_package
            temp(k,:) = reshape(csi_mag{i}{j}{k}', 1, []); 
            temp2(k,:) = reshape(csi_phase{i}{j}{k}', 1, []); 
        end
        csi_mag{i}{j} = temp;
        csi_phase{i}{j} = temp;
    end
end

save csi_mag csi_mag
save csi_phase csi_phase