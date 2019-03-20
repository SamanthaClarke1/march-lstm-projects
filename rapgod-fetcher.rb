#!/usr/bin/ruby

require 'httpclient'


puts "starting up..\n\n"

defUrl = 'http://ohhla.com/anonymous/'
maxArtists = 1
artistOffset = 0
artistAddChance = 1#maxArtists / 3318.0

$clnt = HTTPClient.new()

def sanitizeName(name)
    name = name.downcase
    name = name.strip
    name.tr!('()%!@#$^&*[]{}\\/<>~`\'\":;?\=+.', '')
    name.tr!(' -', '_')
    name.gsub '__', '_'
end

def getPossiblePaths(base, name)
    tnames = []

    h = { 'the' => '', 'and' => ''}
    tnames[0] = base + sanitizeName(name.gsub(/\w+/) { |m| h.fetch(m,m) }) + '/'
    tnames[1] = base + sanitizeName(name) + '/'
    tnames[2] = base + (sanitizeName(name).gsub '_', '') + '/'
    tnames[3] = base + (sanitizeName(name).gsub '1234567890', '') + '/'

    return tnames
end

def getSubDirs(urlSoFar)
    res = $clnt.get(urlSoFar, :follow_redirect => true)

    if res.status == 200 then
        c = res.content
        if c.index('<table>') then
            c = c[(c.index('<table>')+7)..(c.index('</table>'))]
            ca = c.split('</td><td><a href=')

            ca.shift()
            ca = ca.map { |c| c[(c.index('">')+2)..(c.index('</a>')-1)] }
            ca.shift()

            return ca
        end
    end
    return false
end

def getBounty(destFile, urlSoFar)
    res = $clnt.get(urlSoFar, :follow_redirect => true)
    if res.status == 200 then
        c = res.content
        if c.index('<pre>') then
            c = c[(c.index('<pre>')+5)..(c.index('</pre>')-1)]
            return c
        end
    end
    return false
end

def bountyHunt(destFile, urlSoFar, depth=1)
    subDirs = getSubDirs(urlSoFar)
    bounties = []
    
    if subDirs and subDirs.length > 0 then
        subDirs.each { |s|
            #puts urlSoFar + " " + s
            if s.index('.txt') then
                tb = getBounty(destFile, urlSoFar+s)
                if tb != false then
                    #puts tb
                    bounties.push(tb)
                end
            else
                tb = bountyHunt(destFile, urlSoFar+s, depth+1)
                #puts tb[0].to_s + " | " + tb[1].to_s + " | " + tb[2].length.to_s
                bounties.push(tb[2])
                #if tb[0] == '1' then
                #   bounties.push(tb[2])
                #end
            end
        }
        return [1, destFile, bounties]
    else
        return [0, destFile, bounties]
    end
    return [0, destFile, bounties]
end

def getUrls(artistfile, baseUrl, maxArtists, artistOffset, artistAddChance)
    tArtists = []
    lineNum=0
    text=File.open(artistfile).read

	text.each_line do |line|
		if artistOffset > 0 then
			artistOffset -= 1
		elsif lineNum < maxArtists then
			if rand() < artistAddChance then
				tArtists.push(*getPossiblePaths(baseUrl, line))
				lineNum += 1
			end
        end
    end

    return tArtists.uniq
end

def saveSongs(artist, songs)
    i = 0
    songs.each { |s|
        i += 1
        s = s.gsub(/\[[A-z0-9*\s'"+-=!<>\-=_.,\/\\]+\]/i, '')
        s = s.gsub(/^(Verse One: |Intro: |Artist: |Album: |Song: |Typed by: ).+?\n/i, '')
		s = s.gsub(/-=.+?=-\n/, '')
		s = s.tr('¢ã®', '')
        s = s.gsub('nigga', 'ninja')
        s = s.gsub('nigger', 'ninja')
        s = s.gsub('Nigga', 'Ninja')
        s = s.gsub('Nigger', 'Ninja')
        #s = s.sub(/[A-z0-9]+/, '')
        File.open("songs/"+(artist.gsub '/', '')+'.'+i.to_s+".txt", "w+") { |f|
            f.write("\n")
            f.write(s)
        }
        sleep(0.2)
    }
end

def threadRun(urls, defUrl)
    threads = 64
    successes = 0
	fails = 0
    
    for i in 0..((urls.length / threads).floor) do
		ti = i * threads
		threadHandles = Array.new(threads)

        threads.times do |j|
            j = j + ti
            if j < urls.length then
                puts 'sending off :  ' + j.to_s + " j " + urls[j]
                threadHandles[j] = Thread.new {
                    Thread.current['j'] = j
                    Thread.current['bounty'] = bountyHunt((urls[j].gsub defUrl, ''), urls[j])
                }
            end
		end

		threadHandles.each do |thrd|
			if thrd then
				thrd.join
				tb = thrd['bounty']

				#puts tb[0]
				if tb[0] == 1 then
					successes += 1
				else
					fails += 1
				end

				puts tb[1] + ' returned ' + (tb[0] == 1 ? 'successfully' : 'unsuccessfully') + '. ('+successes.to_s+'+'+fails.to_s+'='+(successes+fails).to_s+'/'+urls.length.to_s+')'

				tb[2].flatten!
				puts tb[1] + ' ==> ' + tb[2].length.to_s

				if tb[2].length > 0 then
					saveSongs(tb[1], tb[2])
				end
			end
		end
    end
end

urls = getUrls('artists.txt', defUrl, maxArtists, artistOffset, artistAddChance)

threadRun(urls, defUrl)
#tb = bountyHunt('88rising', 'http://ohhla.com/anonymous/88rising/')
#puts tb.class # is array here
#getSubDirs('http://ohhla.com/anonymous/88rising/head_in/')

print "\nthats all folks"
