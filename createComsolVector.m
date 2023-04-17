function command = createComsolVector(scale, sections, varargin)
% CreateComsolVector - Copy directly to clipboard the comsol command to
% generate the desired range of values
% Given the start, end and step for every frequency section, and having
% decided the scale, the function return directly into the clipboard the
% command for defining a frequency array in comsol. The user just need to
% press ctrl+v to paste it in the program
% inputs:
%   - scale    : string, scale step for the array
%               - lin
%               - log
%   - section : matrix (Nx3) or row array, array of vectors with the format: 
%               [[start_value, end_value, numper_of_points], ...]. The
%               matrix can be created modifying the comma for semicolon 
%               [[start_value, end_value, numper_of_points]; ...]
%   - varargin: 
%        - display : boolean, if true, display a plot to visualize the 
%                    resulting vector (true, false).
%                    default: true
%
% output: 
%   - command : string, Contains the comsol command. 

% Author: Pablo Táboas Rivas    Date: 29/03/2023

valList = [];
lastValue = 0;
display = true;

if ~isempty(varargin)
    display = varargin{1};
end
    
if size(sections, 1) == 1
    sections = reshape(sections,[3,length(sections)/3])';
end
val = string(zeros(size(sections, 1), 1));


for n = 1:size(sections,1)
    a = sections(n,:); % stracting individual zones a = (start, end ,step) 
    
    if strcmp(scale, "lin")
        step = (a(2)-a(1))/(a(3)-1); %(end - start)/step 
        
        % checking if a(1) is already in the final vector
        if lastValue == a(1) 
            tmp = [a(1)+step, step, a(2)];
        else
            tmp = [a(1), step, a(2)];
        end
        % val(n) := "range(start, step, end) "
        val(n) = ['range(' , ...
                    num2str(tmp(1)), ',', ...
                    num2str(tmp(2)), ',', ...
                    num2str(tmp(3)), ') '];
        % Obtaining numerical values for graphical revision
        valList = [valList tmp(1):tmp(2):tmp(3)];
        
    elseif strcmp(scale, "log")
        step = (log10(a(2))- log10(a(1)))/(a(3)-1); % logaritmic step based on Comsol performance
        
        % checking if a(1) is already in the final vector
        if lastValue == a(1)
            tmp = [10^(log10(a(1))+step), step, a(2)];
        else
            tmp = [a(1), step, a(2)];
        end
        % val(n) := "10^range(log10(start), step, log10(end)) "
        val(n) = ['10^(range(', ...
                  'log10(', num2str(tmp(1)), '),', ...
                  num2str(tmp(2)), ',', ... 
                  'log10(', num2str(tmp(3)), ')) '];
        % Obtaining numerical values for graphical revision
        valList = [valList 10.^(log10(tmp(1)):tmp(2):log10(tmp(3)))];
        
    else
        print("Unknown spacing scale. valid: lin/log");
        return 
    end
    % saving last value to compare it with the first of the next vector
    lastValue = a(2); 
end

command = strjoin(val);
clipboard("copy", command)

message = ['Command copied in the clipboard.\n' ...
        'Total number of data points: ',string(length(valList)),'\n', ...
        'Press ctrl+v to paste it \n'];
fprintf(strjoin(message))

if display
    figure(10)
    % logaritmic x axis if the scale is log. Linear otherwise
    if strcmp(scale, "log")
        semilogx(valList, 1:length(valList), "*-");
        grid on;
    else
        plot(valList, 1:length(valList), "*-");
        grid on;
    end
    ylabel("Data point number") 
    xlabel("Data value") 
end
end
