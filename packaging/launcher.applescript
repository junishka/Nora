-- Builder.app launcher — opens a new Terminal window and runs the
-- bundled `builder` binary from the .app's Resources directory.
--
-- Double-clicking Builder.app on macOS runs this script. It resolves
-- the bundled binary path from the .app's own location (so users can
-- move/rename the app without breaking it), opens Terminal, and
-- invokes `builder` without arguments — which triggers the
-- interactive data-dir prompt in app.py's `_prompt_for_cwd`.
--
-- Written as AppleScript rather than a shell script because
-- `osacompile` produces a proper .app bundle from an .applescript
-- file with a single command, and the resulting .app behaves like
-- any other macOS application (appears in Launchpad, Dock, Cmd-Tab,
-- etc.). The Resources/ subdirectory inside the .app is where we
-- stash the PyInstaller bundle.

on run
	set appPath to (POSIX path of (path to me))
	set builderBin to appPath & "Contents/Resources/builder/builder"
	set quotedBin to quoted form of builderBin

	tell application "Terminal"
		activate
		-- `do script` opens a new Terminal window and runs the command.
		-- Quoting the binary path protects against spaces in
		-- /Applications/Builder.app/... (unusual but possible).
		do script quotedBin
	end tell
end run
