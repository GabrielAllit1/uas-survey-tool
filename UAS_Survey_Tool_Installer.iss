; UAS_Survey_Tool_Installer.iss
; Inno Setup script to package the UAS Survey Tool v2.0

[Setup]
AppName=UAS Survey Tool
AppVersion=2.0
DefaultDirName={autopf}\UAS Survey Tool
DefaultGroupName=UAS Survey Tool
OutputDir=dist
OutputBaseFilename=UAS_Survey_Tool_Installer
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
AllowNoIcons=yes
DisableProgramGroupPage=no
LicenseFile=LICENSE.txt
SetupIconFile=app_icon.ico
UninstallIconFile=app_icon.ico
WizardStyle=modern
AppPublisher=Gabriel Allit
AppPublisherURL=https://allituas.com
AppSupportURL=https://allituas.com/support
AppUpdatesURL=https://allituas.com/updates

[Files]
Source: "dist\main\UAS_Survey_Tool.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "styles.qss"; DestDir: "{app}"; Flags: ignoreversion
Source: "app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "{code:GetAnacondaPath}\Library\share\gdal\*"; DestDir: "{app}\gdal_data"; Flags: recursesubdirs
Source: "{code:GetAnacondaPath}\Library\share\proj\*"; DestDir: "{app}\proj"; Flags: recursesubdirs
Source: "{code:GetAnacondaPath}\lib\site-packages\matplotlib\mpl-data\*"; DestDir: "{app}\matplotlib\mpl-data"; Flags: recursesubdirs
Source: "LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\UAS Survey Tool"; Filename: "{app}\UAS_Survey_Tool.exe"
Name: "{group}\Uninstall UAS Survey Tool"; Filename: "{uninstallexe}"
Name: "{userdesktop}\UAS Survey Tool"; Filename: "{app}\UAS_Survey_Tool.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{app}\UAS_Survey_Tool.exe"; Description: "Launch UAS Survey Tool"; Flags: nowait postinstall skipifsilent

[Code]
function GetAnacondaPath(Param: string): string;
begin
  Result := ExpandConstant('{pf}\Anaconda3\envs\uas_survey_tool');
  if not DirExists(Result) then
    Result := ExpandConstant('{userappdata}\Anaconda3\envs\uas_survey_tool');
  if not DirExists(Result) then
    Result := 'C:\Users\gabri\Anaconda3\envs\uas_survey_tool';
end;